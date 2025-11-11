#!/usr/bin/env python3
"""
Q2 – Pairwise ablation to find the best 1–2 time series to predict tops on BTC/ETH/SOL.

Ready-to-run version:
- No args required. If --input-csv is omitted, the script auto-detects CRYPTO_4ASSETS_DAILY.csv
  next to this script or in ./data.
- Outputs are written NEXT TO THIS SCRIPT in: ./outputs_q2/run_YYYYMMDD_HHMMSS/
- Shows absolute output path and (on Windows) opens the folder at the end.

Input CSV expected columns (case-sensitive by default):
- date, BTC, ETH, SOL (XMR optional/ignored here unless you add it to --symbols)

If your columns are named differently (e.g. BTC_close), use --price-col-suffix.
"""
from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, log_loss, mean_squared_error


@dataclass
class Config:
    input_csv: str | None
    date_col: str
    symbols: List[str]
    price_col_suffix: str
    horizon: int
    up_thresh: float
    down_thresh: float
    post_h: int
    n_splits: int
    min_train: int
    test_size: int
    output_dir: str
    use_existing_labels: bool


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Q2 Pairwise ablation for BTC/ETH/SOL (ready-to-run)")
    p.add_argument("--input-csv", default=None, help="Path to merged CSV. If omitted, we look for CRYPTO_4ASSETS_DAILY.csv next to this script or in ./data")
    p.add_argument("--date-col", default="date")
    p.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"], help="Targets and candidate inputs (order matters)")
    p.add_argument("--price-col-suffix", default="", help="Suffix appended to each symbol to find its price column (default empty for CRYPTO_4ASSETS_DAILY.csv)")

    # Labeling heuristic
    p.add_argument("--horizon", type=int, default=45)
    p.add_argument("--up-thresh", type=float, default=0.10)
    p.add_argument("--down-thresh", type=float, default=0.05)
    p.add_argument("--post-h", type=int, default=30)
    p.add_argument("--use-existing-labels", action="store_true", help="Use y_<SYM> columns if present instead of generating labels")

    # CV / splits
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--min-train", type=int, default=365, help="Min train observations before first split")
    p.add_argument("--test-size", type=int, default=120, help="Test size per fold")

    p.add_argument("--output-dir", default="outputs_q2")

    args = p.parse_args()

    return Config(
        input_csv=args.input_csv,
        date_col=args.date_col,
        symbols=args.symbols,
        price_col_suffix=args.price_col_suffix,
        horizon=args.horizon,
        up_thresh=args.up_thresh,
        down_thresh=args.down_thresh,
        post_h=args.post_h,
        n_splits=args.n_splits,
        min_train=args.min_train,
        test_size=args.test_size,
        output_dir=args.output_dir,
        use_existing_labels=args.use_existing_labels,
    )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _find_default_csv() -> str | None:
    script_dir = Path(__file__).parent.resolve()
    candidates = [
        script_dir / "CRYPTO_4ASSETS_DAILY.csv",
        script_dir / "data" / "CRYPTO_4ASSETS_DAILY.csv",
        Path.cwd() / "CRYPTO_4ASSETS_DAILY.csv",
        Path.cwd() / "data" / "CRYPTO_4ASSETS_DAILY.csv",
        script_dir.parent / "CRYPTO_4ASSETS_DAILY.csv",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def build_features(df: pd.DataFrame, symbols: List[str], price_suffix: str) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)
    for sym in symbols:
        c = f"{sym}{price_suffix}"
        if c not in out.columns:
            raise ValueError(f"Missing column: {c}")
        px = out[c].astype(float)
        lr1 = np.log(px).diff(1)
        lr7 = np.log(px).diff(7)
        lr14 = np.log(px).diff(14)
        vol14 = lr1.rolling(14).std()
        mom30 = px.pct_change(30)
        out[f"{sym}_lr1"] = lr1
        out[f"{sym}_lr7"] = lr7
        out[f"{sym}_lr14"] = lr14
        out[f"{sym}_vol14"] = vol14
        out[f"{sym}_mom30"] = mom30
    return out


def generate_top_labels(prices: pd.Series, horizon: int, up_thresh: float, down_thresh: float, post_h: int) -> pd.Series:
    px = prices.astype(float).values
    n = len(px)
    y = np.zeros(n, dtype=float)
    for t in range(n):
        end = min(n, t + horizon + 1)
        if end <= t + 1:
            continue
        window = px[t+1:end]
        if window.size == 0:
            continue
        fmax_idx = int(np.argmax(window))
        fmax = window[fmax_idx]
        if px[t] <= 0:
            continue
        up = (fmax / px[t]) - 1.0
        if up < up_thresh:
            continue
        post_start = t + 1 + fmax_idx
        post_end = min(n, post_start + post_h + 1)
        if post_end <= post_start + 1:
            continue
        post_window = px[post_start:post_end]
        fmin = np.min(post_window) if post_window.size else fmax
        dd = 1.0 - (fmin / fmax)
        if dd >= down_thresh:
            y[t] = 1.0
    return pd.Series(y, index=prices.index, name="label")


def maybe_add_labels(df: pd.DataFrame, date_col: str, symbols: List[str], price_suffix: str, horizon: int, up_t: float, down_t: float, post_h: int, use_existing: bool) -> pd.DataFrame:
    out = df.copy()
    if use_existing:
        for sym in symbols:
            ycol = f"y_{sym}"
            if ycol not in out.columns:
                pcol = f"{sym}{price_suffix}"
                out[ycol] = generate_top_labels(out[pcol], horizon, up_t, down_t, post_h)
        return out
    for sym in symbols:
        pcol = f"{sym}{price_suffix}"
        out[f"y_{sym}"] = generate_top_labels(out[pcol], horizon, up_t, down_t, post_h)
    return out


def timeseries_folds(n: int, n_splits: int, min_train: int, test_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    folds = []
    start_train_end = min_train
    while len(folds) < n_splits:
        train_end = start_train_end + len(folds) * test_size
        test_end = train_end + test_size
        if test_end > n:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        folds.append((train_idx, test_idx))
    return folds


def fit_eval(X: pd.DataFrame, y: pd.Series, dates: pd.Series, cv_cfg: Dict, run_tag: str, outdir: Path) -> Dict[str, float]:
    scaler = StandardScaler()
    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)

    ap_list, ll_list, br_list = [], [], []
    preds_all = []

    folds = timeseries_folds(len(X), cv_cfg["n_splits"], cv_cfg["min_train"], cv_cfg["test_size"])
    if len(folds) == 0:
        raise ValueError("Not enough data to make even one fold. Increase data or adjust min-train/test-size.")

    for i, (tr, te) in enumerate(folds, 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        dte = dates.iloc[te]

        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        clf.fit(Xtr_s, ytr)
        p = clf.predict_proba(Xte_s)[:, 1]

        ap = average_precision_score(yte, p)
        eps = 1e-9
        p_safe = np.clip(p, eps, 1 - eps)
        ll = log_loss(yte, p_safe)
        br = mean_squared_error(yte, p)

        ap_list.append(ap)
        ll_list.append(ll)
        br_list.append(br)

        fold_df = pd.DataFrame({"date": dte.values, "y": yte.values, "p": p, "fold": i})
        preds_all.append(fold_df)

    preds_df = pd.concat(preds_all, ignore_index=True)
    preds_path = outdir / f"preds_{run_tag}.csv"
    preds_df.to_csv(preds_path, index=False)

    return {
        "AP_mean": float(np.mean(ap_list)),
        "AP_std": float(np.std(ap_list)),
        "LogLoss_mean": float(np.mean(ll_list)),
        "LogLoss_std": float(np.std(ll_list)),
        "Brier_mean": float(np.mean(br_list)),
        "Brier_std": float(np.std(br_list)),
        **{f"AP_fold{i}": v for i, v in enumerate(ap_list, 1)},
        **{f"LogLoss_fold{i}": v for i, v in enumerate(ll_list, 1)},
        **{f"Brier_fold{i}": v for i, v in enumerate(br_list, 1)},
        "preds_path": str(preds_path.resolve()),
    }


def main():
    cfg = parse_args()

    from datetime import datetime
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    script_dir = Path(__file__).parent.resolve()
    base_out = Path(cfg.output_dir)
    if not base_out.is_absolute():
        base_out = script_dir / base_out
    outdir = base_out / run_name
    ensure_dir(str(outdir))

    detected_csv = cfg.input_csv or _find_default_csv()
    if detected_csv is None:
        print("[Q2] Aucun --input-csv fourni et CRYPTO_4ASSETS_DAILY.csv introuvable. Place le fichier à côté du script (ou dans ./data) OU passe --input-csv.")
        raise SystemExit(2)

    df = pd.read_csv(detected_csv)
    if cfg.date_col not in df.columns:
        raise ValueError(f"Missing date column '{cfg.date_col}' in input CSV")
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df = df.sort_values(cfg.date_col).reset_index(drop=True)
    df = df.rename(columns={cfg.date_col: "date"})

    df = build_features(df, cfg.symbols, cfg.price_col_suffix)
    df = maybe_add_labels(df, "date", cfg.symbols, cfg.price_col_suffix,
                          cfg.horizon, cfg.up_thresh, cfg.down_thresh, cfg.post_h,
                          cfg.use_existing_labels)

    results = []

    for target in cfg.symbols:
        others = [s for s in cfg.symbols if s != target]
        if len(others) < 2:
            raise ValueError("Need at least 3 symbols to evaluate singletons and pairs")
        combos = [(others[0],), (others[1],), tuple(others)]

        for combo in combos:
            feat_cols = []
            for sym in combo:
                feat_cols += [
                    f"{sym}_lr1", f"{sym}_lr7", f"{sym}_lr14",
                    f"{sym}_vol14", f"{sym}_mom30",
                ]
            sub = df[["date"] + feat_cols + [f"y_{target}"]].dropna().reset_index(drop=True)
            if sub.empty:
                print(f"[WARN] No data after dropna for target={target} inputs={'+'.join(combo)}. Skipping.")
                continue
            X = sub[feat_cols]
            y = sub[f"y_{target}"]
            dates = sub["date"]

            run_tag = f"{target}_from_{'-'.join(combo)}"
            run_outdir = outdir / target
            ensure_dir(str(run_outdir))

            metrics = fit_eval(
                X, y, dates,
                {"n_splits": cfg.n_splits, "min_train": cfg.min_train, "test_size": cfg.test_size},
                run_tag, run_outdir)

            results.append({"target": target, "inputs": "+".join(combo), **metrics})
            print(f"Done: target={target} inputs={'+'.join(combo)} AP={metrics['AP_mean']:.4f} LogLoss={metrics['LogLoss_mean']:.4f} Brier={metrics['Brier_mean']:.4f}")

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(["target", "AP_mean", "LogLoss_mean"], ascending=[True, False, True])

    out_csv = outdir / "q2_pairwise_results.csv"
    res_df.to_csv(out_csv, index=False)

    md_lines = [
        "# Q2 Pairwise Ablation Results\\n",
        f"Input CSV: {Path(detected_csv).resolve()}\\n\\n",
    ]
    for tgt, grp in res_df.groupby("target"):
        md_lines.append(f"## Target: {tgt}\\n")
        for _, row in grp.iterrows():
            md_lines.append(
                f"- Inputs: {row['inputs']:<10} | AP={row['AP_mean']:.4f} ±{row['AP_std']:.4f} | "
                f"LogLoss={row['LogLoss_mean']:.4f} ±{row['LogLoss_std']:.4f} | "
                f"Brier={row['Brier_mean']:.4f} ±{row['Brier_std']:.4f} | preds: {row['preds_path']}\\n"
            )
        md_lines.append("\\n")

    readme_path = outdir / "README_results.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("".join(md_lines))

    print("\\nSaved:")
    print(f" - Output dir: {outdir.resolve()}")
    print(f" - Summary CSV: {out_csv.resolve()}")
    print(f" - Markdown report: {readme_path.resolve()}")
    print(f" - Per-run predictions under: {outdir.resolve()}\\<TARGET>\\preds_*")

    try:
        if os.name == "nt":
            os.startfile(str(outdir))
    except Exception:
        pass


if __name__ == "__main__":
    main()
