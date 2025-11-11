
import numpy as np, pandas as pd, torch, torch.nn as nn, json, argparse
from sklearn.metrics import average_precision_score, log_loss, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from functools import reduce
from pathlib import Path
import re, sys

# -----------------------
# Config via CLI with sane defaults (same folder as script)
# -----------------------
def get_args():
    p = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    p.add_argument("--btc_csv", type=str, default=str(here/"BINANCE_BTCUSDT_1D_merged.csv"))
    p.add_argument("--eth_csv", type=str, default=str(here/"BINANCE_ETHUSDT_1D_merged.csv"))
    p.add_argument("--sol_csv", type=str, default=str(here/"BINANCE_SOLUSDT_1D_merged.csv"))
    p.add_argument("--xmr_csv", type=str, default=str(here/"KRAKEN_XMRUSD_1D_merged_full.csv"))
    p.add_argument("--out_dir", type=str, default=str(here))
    # Top definition (daily)
    p.add_argument("--K_past", type=int, default=60)
    p.add_argument("--K_future", type=int, default=15)
    p.add_argument("--M", type=int, default=120)
    p.add_argument("--drawdown", type=float, default=0.25)
    # Alert horizon & window lengths
    p.add_argument("--H", type=int, default=45)
    p.add_argument("--L", type=int, default=120)     # shorter for robustness
    p.add_argument("--roll_norm", type=int, default=90)
    # Training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--pos_weight", type=float, default=5.0)
    p.add_argument("--splits", type=int, default=5)
    return p.parse_args()

# -----------------------
# Robust CSV reading with date parsing
# -----------------------
_date_ddmmyyyy = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{4}$")

def _parse_dates(series):
    # Try to detect dd/mm/yyyy strings
    if series.dtype == object:
        sample = str(series.dropna().astype(str).head(1).values[0]) if series.dropna().size else ""
        dayfirst = bool(_date_ddmmyyyy.match(sample))
        dt = pd.to_datetime(series, errors="coerce", utc=True, dayfirst=dayfirst)
        if dt.notna().sum() > 0:
            return dt
    # Try numeric epoch
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.to_datetime(series, errors="coerce", utc=True)
    med = float(np.nanmedian(s.values))
    if med > 1e16:   unit="ns"
    elif med > 1e13: unit="us"
    elif med > 1e11: unit="ms"
    elif med > 1e8:  unit="s"
    else:            unit="s"
    return pd.to_datetime(s, unit=unit, errors="coerce", utc=True)

def robust_read(path):
    # read CSV or ; separated
    raw = pd.read_csv(path)
    if raw.shape[1] == 1 and ";" in str(raw.columns[0]):
        raw = pd.read_csv(path, sep=";")
    raw.columns = [c.strip().lower().replace(" ","_") for c in raw.columns]
    # choose a date-like column
    date_cols = [c for c in raw.columns if any(k in c for k in ["date","time","timestamp","open_time","close_time"])]
    if not date_cols:
        raise FileNotFoundError(f"No date column in {path}")
    parsed = None; chosen=None
    for c in date_cols:
        dt = _parse_dates(raw[c])
        if dt.notna().sum() > 0:
            parsed = dt; chosen = c; break
    if parsed is None:
        raise ValueError(f"Could not parse dates in {path}")
    df = raw.copy()
    # Replace/insert 'date' (naive UTC)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    df.insert(0, "date", parsed.dt.tz_convert("UTC").dt.tz_localize(None))
    if chosen != "date" and chosen in df.columns:
        df = df.drop(columns=[chosen])
    # Map OHLCV
    mapping = {}
    for c in list(df.columns):
        if c == "date": continue
        cl = c.lower()
        if cl in ["open","high","low","close","volume","vwap","trades","quote_volume"]:
            mapping[c] = cl
        elif cl in ["o"]: mapping[c] = "open"
        elif cl in ["h"]: mapping[c] = "high"
        elif cl in ["l"]: mapping[c] = "low"
        elif cl in ["c"]: mapping[c] = "close"
        elif "vwap" in cl: mapping[c] = "vwap"
        elif "trades" in cl or "count" in cl: mapping[c] = "trades"
        elif "quote" in cl and "volume" in cl: mapping[c] = "quote_volume"
        elif "volume" in cl and "quote" not in cl: mapping[c] = "volume"
    if mapping:
        df = df.rename(columns=mapping)
    keep = ["date"]
    for k in ["open","high","low","close","volume"]:
        if k in df.columns: keep.append(k)
    out = df[keep].sort_values("date").reset_index(drop=True)
    for c in out.columns:
        if c != "date":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["date"] = pd.to_datetime(out["date"])
    return out

# -----------------------
# Labels: tops & "top soon"
# -----------------------
def mark_tops(P, Kp=60, Kf=15, M=120, dd=0.25):
    P = np.asarray(P); n=len(P); tops = np.zeros(n, dtype=bool)
    for t in range(Kp, n-max(Kf,M)):
        window = P[t-Kp:t+Kf+1]
        if np.isnan(window).any(): 
            continue
        if not np.isnan(P[t]) and P[t] == np.nanmax(window):
            future_min = np.nanmin(P[t:t+M+1])
            if not np.isnan(future_min) and future_min <= (1-dd)*P[t]:
                tops[t] = True
    return tops

def future_top_label(tops, H=45):
    n=len(tops); Y = np.zeros(n, dtype=int)
    for t in range(n-H):
        Y[t] = int(tops[t:t+H+1].any())
    return Y

# -----------------------
# Windowing
# -----------------------
def windowize(Xdf, Ydf, L, step=1):
    Xv = Xdf.values; Yv = Ydf.values
    Xmat, Ymat, idx = [], [], []
    for t in range(L-1, len(Xv), step):
        block = Xv[t-L+1:t+1, :]
        # Assume we've cleaned NaNs; but still guard
        if np.isnan(block).any():
            continue
        Xmat.append(block)
        Ymat.append(Yv[t, :])
        idx.append(Xdf.index[t])
    return np.array(Xmat, dtype=np.float32), np.array(Ymat, dtype=np.float32), pd.Index(idx)

# -----------------------
# Model
# -----------------------
class LSTMMultiBinary(nn.Module):
    def __init__(self, m, hidden=64, layers=1, dropout=0.1, out_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(m, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers>1 else 0.)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, X):
        out, _ = self.lstm(X)
        hL = out[:, -1, :]
        return self.head(hL)

def train_eval(Xseq, Yseq, idx_seq, with_xmr, cols, cfg):
    # Decide which columns to keep
    use_cols = list(cols)
    if not with_xmr:
        use_cols = [c for c in use_cols if ("_XMR" not in c and not c.endswith("XMR"))]
    col_index = [list(cols).index(c) for c in use_cols]
    # Short-circuit if we have no sequences
    if Xseq.shape[0] == 0:
        raise RuntimeError(
            "Aucune séquence construite (Xseq est vide). "
            "Réduis --L (ex: 120), assure-toi que les 4 séries ont des dates communes, "
            "et vérifie la normalisation rolling (--roll_norm)."
        )
    xi = Xseq[..., col_index]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tscv = TimeSeriesSplit(n_splits=cfg["splits"])
    metrics = {a: dict(ap=[], ll=[], brier=[]) for a in ['BTC','ETH','SOL']}
    preds = []

    for tr, te in tscv.split(xi):
        Xtr = torch.tensor(xi[tr]).to(device)
        Xte = torch.tensor(xi[te]).to(device)
        Ytr = torch.tensor(Yseq[tr]).to(device)
        Yte = torch.tensor(Yseq[te]).to(device)

        model = LSTMMultiBinary(m=xi.shape[-1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg["pos_weight"]]*3, device=device))

        model.train()
        for _ in range(cfg["epochs"]):
            idx = torch.randperm(len(Xtr), device=device)
            for k in range(0, len(idx), cfg["batch"]):
                b = idx[k:k+cfg["batch"]]
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

        fold_df = pd.DataFrame({
            "date": idx_seq[te].astype("datetime64[ns]"),
            "p_BTC": probs[:,0], "p_ETH": probs[:,1], "p_SOL": probs[:,2],
            "y_BTC": ytrue[:,0], "y_ETH": ytrue[:,1], "y_SOL": ytrue[:,2],
        })
        preds.append(fold_df)

        for j, a in enumerate(['BTC','ETH','SOL']):
            eps=1e-6
            pj = np.clip(probs[:,j], eps, 1-eps)
            yj = ytrue[:,j]
            try:
                ap = average_precision_score(yj, pj)
            except ValueError:
                ap = np.nan
            ll = log_loss(yj, pj, labels=[0,1])
            br = brier_score_loss(yj, pj)
            metrics[a]['ap'].append(ap)
            metrics[a]['ll'].append(ll)
            metrics[a]['brier'].append(br)

    out = {a:{k: float(np.nanmean(v)) for k,v in d.items()} for a,d in metrics.items()}
    return out, pd.concat(preds, ignore_index=True).sort_values("date")

# -----------------------
# Main
# -----------------------
def main():
    args = get_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load & align closes
    btc = robust_read(args.btc_csv)[["date","close"]].rename(columns={"close":"BTC"})
    eth = robust_read(args.eth_csv)[["date","close"]].rename(columns={"close":"ETH"})
    sol = robust_read(args.sol_csv)[["date","close"]].rename(columns={"close":"SOL"})
    xmr = robust_read(args.xmr_csv)[["date","close"]].rename(columns={"close":"XMR"})

    merged = reduce(lambda l,r: pd.merge(l,r,on="date",how="outer"),
                    [btc, eth, sol, xmr]).sort_values("date").reset_index(drop=True)

    # Garder uniquement les dates où TOUTES les closes existent
    mi = merged.set_index("date")
    all_present = mi[["BTC","ETH","SOL","XMR"]].notna().all(axis=1)
    mi = mi.loc[all_present].copy()
    if mi.shape[0] < (args.roll_norm + args.L + args.H + 10):
        print("[WARN] Peu de lignes après filtrage des dates communes. "
              "Réduisez L/roll_norm ou assurez-vous que les 4 séries se chevauchent suffisamment.", file=sys.stderr)

    # 2) Features (sur dates communes)
    feat = pd.DataFrame(index=mi.index)
    for sym in ["BTC","ETH","SOL","XMR"]:
        r = np.log(mi[sym]).diff()
        feat[f"r_{sym}"] = r
        feat[f"vol14_{sym}"] = r.rolling(14).std()
        feat[f"mom30_{sym}"] = r.rolling(30).sum()

    # 3) Labels (tops) sur mêmes dates
    labels = {}
    for a in ["BTC","ETH","SOL"]:
        tops_a = mark_tops(mi[a].values, Kp=args.K_past, Kf=args.K_future, M=args.M, dd=args.drawdown)
        labels[a] = pd.Series(future_top_label(tops_a, H=args.H), index=mi.index).astype(int)
    Y = pd.concat(labels, axis=1).dropna().astype(int)

    # --- Align features & labels robustly ---
    feat.index = pd.to_datetime(feat.index).tz_localize(None)
    Y.index    = pd.to_datetime(Y.index).tz_localize(None)
    common = feat.index.intersection(Y.index).sort_values()
    X = feat.loc[common]
    Y = Y.loc[common]

    # 4) Rolling z-score (sans fuite) + drop des lignes NaN
    Xz = X.copy()
    win = int(args.roll_norm)
    for col in Xz.columns:
        mu = Xz[col].rolling(win, min_periods=win).mean()
        sd = Xz[col].rolling(win, min_periods=win).std()
        Xz[col] = (Xz[col]-mu)/sd
    Xz = Xz.replace([np.inf, -np.inf], np.nan).dropna()

    # 5) Final alignment again after rolling drop
    Y = Y.loc[Xz.index]

    # 6) Windowize
    L = int(args.L)
    Xseq, Yseq, idx_seq = windowize(Xz, Y, L=L, step=1)
    cols = Xz.columns

    # 7) Train & evaluate (with/without XMR)
    cfg = dict(epochs=args.epochs, batch=args.batch, lr=args.lr, weight_decay=args.weight_decay,
               pos_weight=args.pos_weight, splits=args.splits)
    met0, pred0 = train_eval(Xseq, Yseq, idx_seq, with_xmr=False, cols=cols, cfg=cfg)
    met1, pred1 = train_eval(Xseq, Yseq, idx_seq, with_xmr=True,  cols=cols, cfg=cfg)

    # 8) Save outputs
    merged_out = mi.reset_index(names="date")[["date","BTC","ETH","SOL","XMR"]]
    merged_out.to_csv(out_dir/"CRYPTO_4ASSETS_DAILY.csv", index=False)
    pred0.to_csv(out_dir/"probas_without_xmr.csv", index=False)
    pred1.to_csv(out_dir/"probas_with_xmr.csv", index=False)
    with open(out_dir/"metrics_without_xmr.json","w",encoding="utf-8") as f:
        json.dump(met0, f, indent=2)
    with open(out_dir/"metrics_with_xmr.json","w",encoding="utf-8") as f:
        json.dump(met1, f, indent=2)

    # 9) Print quick summary
    summary = {
        "X_shape": list(Xseq.shape),
        "Y_shape": list(Yseq.shape),
        "first_date_seq": str(idx_seq.min().date()) if len(idx_seq) else None,
        "last_date_seq": str(idx_seq.max().date()) if len(idx_seq) else None,
        "metrics_without_xmr": met0,
        "metrics_with_xmr": met1,
        "out_files": {
            "dataset": str(out_dir/"CRYPTO_4ASSETS_DAILY.csv"),
            "probas_without_xmr": str(out_dir/"probas_without_xmr.csv"),
            "probas_with_xmr": str(out_dir/"probas_with_xmr.csv"),
            "metrics_without_xmr": str(out_dir/"metrics_without_xmr.json"),
            "metrics_with_xmr": str(out_dir/"metrics_with_xmr.json"),
        }
    }
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
