import argparse, os, numpy as np, pandas as pd, torch
from pathlib import Path
from src.data.preprocess import load_fd, add_rul_labels, SENSOR_COLS
from src.data.graph_builder import build_sensor_graph
from sklearn.model_selection import train_test_split

def windowize(df, L, stride, mu, sd):
    xs, ys, units = [], [], []
    for uid, d in df.groupby("unit"):
        A = d[SENSOR_COLS].values.astype(np.float32)                  # [T,N]
        A = (A - mu) / sd
        r = d["RUL"].values.astype(np.float32)
        for t in range(L, len(d)+1, stride):
            xs.append(A[t-L:t][...,None])                             # [L,N,1]
            ys.append(r[t-1])
            units.append(uid)
    X = np.stack(xs, axis=0) if xs else np.empty((0,L,len(SENSOR_COLS),1), np.float32)
    y = np.array(ys, dtype=np.float32)
    return X, y, np.array(units)

def main(a):
    tr, te, _ = load_fd(f"data/CMAPSS/{a.fd}", a.fd)
    tr = add_rul_labels(tr, cap=a.cap)
    # normalization from TRAIN only
    mu = tr[SENSOR_COLS].values.astype(np.float32).mean(0, keepdims=True)
    sd = tr[SENSOR_COLS].values.astype(np.float32).std(0, ddof=1, keepdims=True)
    sd[sd==0] = 1.0

    # build graph (returns edge_index and weights)
    try:
        edge_index, edge_weight = build_sensor_graph(tr, SENSOR_COLS, thresh=a.corr_thresh, topk=None)
    except TypeError:
        # older version returning only edge_index
        edge_index = build_sensor_graph(tr, SENSOR_COLS, thresh=a.corr_thresh)
        edge_weight = None

    # windows
    X, y, units = windowize(tr, a.L, a.stride, mu, sd)

    # unit-wise split for validation
    u_train, u_val = train_test_split(np.unique(units), test_size=a.val_split, random_state=a.seed)
    tr_idx = np.where(np.isin(units, u_train))[0]
    va_idx = np.where(np.isin(units, u_val))[0]

    cache = {
        "X": torch.tensor(X, dtype=torch.float32),
        "y": torch.tensor(y, dtype=torch.float32),
        "train_idx": torch.tensor(tr_idx, dtype=torch.long),
        "val_idx": torch.tensor(va_idx, dtype=torch.long),
        "edge_index": edge_index.long(),
        "norm": {"mu": mu.astype(np.float32)[None,...], "sd": sd.astype(np.float32)[None,...]},
        "meta": {"fd": a.fd, "L": a.L, "stride": a.stride, "cap": a.cap, "corr_thresh": a.corr_thresh,
                 "val_split": a.val_split, "seed": a.seed},
    }
    if edge_weight is not None:
        cache["edge_weight"] = edge_weight.float()

    outdir = Path("data/cache"); outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{a.fd.lower()}_L{a.L}_s{a.stride}_cache.pt"
    torch.save(cache, outdir / fname)
    print(f"Saved cache -> {outdir/fname} | X {tuple(cache['X'].shape)} y {tuple(cache['y'].shape)}")

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fd", default="FD001")
    p.add_argument("--L", type=int, default=50)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--cap", type=int, default=130)
    p.add_argument("--corr_thresh", type=float, default=0.3)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
