# =========================
# Config & imports
# =========================
import numpy as np, pandas as pd, math, torch, torch.nn as nn
from sklearn.metrics import average_precision_score, log_loss, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

CFG = dict(
    K_past=60,      # jours de passé pour détecter le max local (~3 mois)
    K_future=15,    # jours de tolérance autour du pic
    M=120,          # fenêtre pour vérifier le drawdown après le pic (~6 mois)
    drawdown=0.25,  # 25% mini
    H=45,           # horizon d’alerte "un top arrive dans <= H jours"
    L=180,          # longueur de fenêtre pour l’entrée LSTM (~6 mois)
    step=1,         # stride de fenêtrage
    hidden=64, layers=1, dropout=0.1,
    epochs=15, batch=256, lr=1e-3, weight_decay=1e-5,
    pos_weight=5.0,         # pondération des rares positifs
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# =========================
# 1) Chargement & prépa basique
# =========================
df = pd.read_csv('prices.csv', parse_dates=['date']).set_index('date').sort_index()
assert set(['BTC','ETH','SOL','XMR']).issubset(df.columns)
assets = ['BTC','ETH','SOL','XMR']

# Log-returns (évite l’échelle), + petites features simples
rets = np.log(df[assets]).diff()
feat = pd.DataFrame(index=df.index)
for a in assets:
    feat[f'r_{a}']  = rets[a]
    feat[f'vol_{a}'] = rets[a].rolling(14).std()
    feat[f'mom_{a}'] = rets[a].rolling(30).sum()
feat = feat.dropna()

# =========================
# 2) Détection des TOPS & labels "top bientôt"
# =========================
def mark_tops(P, Kp, Kf, M, dd):
    """Renvoie un booléen par jour: True si c'est un top au sens (i)+(ii)."""
    P = P.values; n=len(P)
    tops = np.zeros(n, dtype=bool)
    for t in range(Kp, n-max(Kf,M)):
        window = P[t-Kp:t+Kf+1]
        if P[t] == window.max() and P[t:t+M+1].min() <= (1-dd)*P[t]:
            tops[t] = True
    return tops

def future_top_label(tops, H):
    """Y_t = 1 si un top survient dans la fenêtre [t, t+H]."""
    n=len(tops); Y = np.zeros(n, dtype=int)
    for t in range(n-H):
        Y[t] = int(tops[t:t+H+1].any())
    return Y

labels = {}
for a in ['BTC','ETH','SOL']:
    tops_a = mark_tops(df[a].reindex(feat.index), CFG['K_past'], CFG['K_future'], CFG['M'], CFG['drawdown'])
    Y_a = future_top_label(tops_a, CFG['H'])
    y = pd.Series(Y_a, index=feat.index).astype(int)
    labels[a] = y

Y = pd.concat(labels, axis=1).dropna().astype(int)  # colonnes: BTC, ETH, SOL
X = feat.loc[Y.index]

# =========================
# 3) Normalisation rolling (pas de fuite)
# =========================
def rolling_zscore(frame, win=252):
    mu = frame.rolling(win, min_periods=win).mean()
    sd = frame.rolling(win, min_periods=win).std().replace(0, np.nan)
    Z  = (frame - mu) / sd
    return Z

Xz = rolling_zscore(X).dropna()
Y  = Y.loc[Xz.index]

# =========================
# 4) Fenêtrage en séquences pour LSTM
# =========================
def windowize(Xdf, Ydf, L, step=1):
    Xmat, Ymat, idx = [], [], []
    Xv = Xdf.values; Yv = Ydf.values
    for t in range(L-1, len(Xv), step):
        Xmat.append(Xv[t-L+1:t+1, :])    # [L, m]
        Ymat.append(Yv[t, :])            # [3]
        idx.append(Xdf.index[t])
    return np.array(Xmat, dtype=np.float32), np.array(Ymat, dtype=np.float32), pd.Index(idx)

Xseq, Yseq, idx_seq = windowize(Xz, Y, CFG['L'], CFG['step'])
m = Xseq.shape[-1]

# =========================
# 5) Modèle LSTM multi-sorties (BTC/ETH/SOL)
# =========================
class LSTMMultiBinary(nn.Module):
    def __init__(self, m, hidden=64, layers=1, dropout=0.1, out_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(m, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers>1 else 0.)
        self.head = nn.Linear(hidden, out_dim)  # logits
    def forward(self, X):
        out, _ = self.lstm(X)           # [B,L,H]
        hL = out[:, -1, :]              # [B,H]
        logits = self.head(hL)          # [B,3]
        return logits

# =========================
# 6) Entraînement + évaluation (walk-forward CV)
# =========================
def train_eval(Xseq, Yseq, with_xmr=True):
    # Ablation XMR: on enlève ses colonnes de features si with_xmr=False
    cols = Xz.columns
    if not with_xmr:
        keep = [c for c in cols if not c.endswith('_XMR')]
        Xi = Xseq[..., [list(cols).index(c) for c in keep]]
        m_eff = Xi.shape[-1]
    else:
        Xi = Xseq; m_eff = Xseq.shape[-1]

    device = CFG['device']
    tscv = TimeSeriesSplit(n_splits=5)
    metrics = {a: dict(ap=[], ll=[], brier=[]) for a in ['BTC','ETH','SOL']}

    for tr, te in tscv.split(Xi):
        Xtr, Xte = torch.tensor(Xi[tr]).to(device), torch.tensor(Xi[te]).to(device)
        Ytr, Yte = torch.tensor(Yseq[tr]).to(device), torch.tensor(Yseq[te]).to(device)

        model = LSTMMultiBinary(m_eff, CFG['hidden'], CFG['layers'], CFG['dropout']).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        pos_w = torch.tensor([CFG['pos_weight']]*3, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        model.train()
        for _ in range(CFG['epochs']):
            # simple mini-batch
            idx = torch.randperm(len(Xtr), device=device)
            for k in range(0, len(idx), CFG['batch']):
                b = idx[k:k+CFG['batch']]
                logits = model(Xtr[b])
                loss = criterion(logits, Ytr[b])
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(Xte).cpu().numpy()
            probs  = 1/(1+np.exp(-logits))
            ytrue  = Yte.cpu().numpy()

        # métriques par actif
        for j, a in enumerate(['BTC','ETH','SOL']):
            # éviter valeurs constantes pour log_loss
            eps = 1e-6
            pj = np.clip(probs[:, j], eps, 1-eps)
            yj = ytrue[:, j]
            try:
                ap   = average_precision_score(yj, pj)
            except ValueError:
                ap = np.nan
            ll   = log_loss(yj, pj, labels=[0,1])
            br   = brier_score_loss(yj, pj)
            metrics[a]['ap'].append(ap)
            metrics[a]['ll'].append(ll)
            metrics[a]['brier'].append(br)

    # moyenne des folds
    out = {a:{k: float(np.nanmean(v)) for k,v in d.items()} for a,d in metrics.items()}
    out['__m_eff__'] = m_eff
    return out

res_without = train_eval(Xseq, Yseq, with_xmr=False)
res_with    = train_eval(Xseq, Yseq, with_xmr=True)

print("==== Sans XMR ====");  print(res_without)
print("==== Avec XMR ====");  print(res_with)

# Lecture rapide:
# - AUC-PR (ap) ↑, LogLoss (ll) ↓, Brier ↓  -> XMR apporte de l'info.
