import argparse, os, numpy as np, torch
from src.data.preprocess import load_fd, add_rul_labels, SENSOR_COLS
from src.data.graph_builder import build_sensor_graph

def build_windows(df_with_rul, L=50, stride=1):
    X, y, units = [], [], []
    for uid, dfu in df_with_rul.groupby("unit"):
        if len(dfu) < L: continue
        x = dfu[SENSOR_COLS].values.astype(np.float32)
        r = dfu["RUL"].values.astype(np.float32)
        for s in range(0, len(dfu)-L+1, stride):
            X.append(x[s:s+L]); y.append(r[s+L-1]); units.append(uid)
    return np.array(X), np.array(y).astype(np.float32), np.array(units)

def zscore_fit(df):
    mu = df[SENSOR_COLS].mean().values.astype(np.float32)
    sd = df[SENSOR_COLS].std(ddof=0).replace(0, 1.0).values.astype(np.float32)
    return mu, sd

def main(a):
    os.makedirs("data/cache", exist_ok=True)
    tr, te, test_rul = load_fd(f"data/CMAPSS/{a.fd}", a.fd)
    tr = add_rul_labels(tr, cap=a.cap)
    mu, sd = zscore_fit(tr)
    X, y, units = build_windows(tr, L=a.L, stride=a.stride)
    Xn = ((X - mu)/sd)[..., None]  # [B,T,N,1]
    edge_index = build_sensor_graph(tr, SENSOR_COLS, thresh=a.corr_thresh).cpu()

    # split by unit (no leakage)
    rng = np.random.default_rng(a.seed)
    uniq = np.unique(units); rng.shuffle(uniq); n_val = int(len(uniq)*a.val_split)
    val_units = set(uniq[:n_val])
    tr_idx = np.array([i for i,u in enumerate(units) if u not in val_units])
    va_idx = np.array([i for i,u in enumerate(units) if u in val_units])

    cache = {"X": Xn.astype(np.float32), "y": y, "units": units,
             "train_idx": tr_idx, "val_idx": va_idx,
             "norm": {"mu": mu, "sd": sd}, "edge_index": edge_index,
             "meta": {"fd": a.fd, "L": a.L, "stride": a.stride, "cap": a.cap,
                      "corr_thresh": a.corr_thresh, "val_units": sorted(list(val_units))}}

    out = f"data/cache/{a.fd.lower()}_L{a.L}_s{a.stride}_cache.pt"
    torch.save(cache, out)
    print("Saved cache ->", out, "| X", cache["X"].shape, "y", cache["y"].shape)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fd", default="FD001")
    p.add_argument("--L", type=int, default=50)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--cap", type=int, default=130)
    p.add_argument("--corr_thresh", type=float, default=0.3)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
