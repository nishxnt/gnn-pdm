import argparse, os, json, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from src.data.preprocess import load_fd, add_rul_labels, SENSOR_COLS
from src.data.graph_builder import build_sensor_graph
from src.utils.metrics import summarise_metrics
from src.models.lstm_reg import LSTMReg
from src.models.gnn.sage_gru import SAGEGRU

def last_windows_for_test(df_test, L):
    items = []
    for uid, dfu in df_test.groupby("unit"):
        if len(dfu) < L: continue
        x = dfu.tail(L)[SENSOR_COLS].values.astype(np.float32)  # [L,N]
        items.append((uid, x[..., None]))                        # add feature dim
    items.sort(key=lambda t: t[0])
    units = [u for u,_ in items]
    X = np.stack([x for _,x in items], axis=0)                   # [U,L,N,1]
    return units, X

@torch.no_grad()
def predict_lstm(cache, X, outdir, hidden=128, layers=2, dropout=0.1):
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    m=LSTMReg(n_in=X.shape[2], hidden=hidden, layers=layers, dropout=dropout).to(dev)
    ck = f"{outdir}/best.ckpt"
    assert os.path.exists(ck), f"Checkpoint not found: {ck}"
    m.load_state_dict(torch.load(ck, map_location=dev)['model'])
    X = torch.tensor(X, dtype=torch.float32).to(dev)            # [U,L,N,1]
    X = X.squeeze(-1)                                           # LSTM expects [B,T,N]
    yhat = m(X).cpu().numpy().reshape(-1)
    return yhat

@torch.no_grad()
def predict_gnn(cache, X, outdir, hg=64, ht=128, layers=2, dropout=0.1):
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    m=SAGEGRU(in_feats=1, hidden_g=hg, hidden_t=ht, layers=layers, dropout=dropout).to(dev)
    ck = f"{outdir}/best.ckpt"
    assert os.path.exists(ck), f"Checkpoint not found: {ck}"
    m.load_state_dict(torch.load(ck, map_location=dev)['model'])
    ei = cache['edge_index'].to(dev)
    X = torch.tensor(X, dtype=torch.float32).to(dev)             # [U,L,N,1]
    yh=[]
    for i in range(X.shape[0]):
        yh.append(m(X[i:i+1], ei).cpu().numpy().reshape(-1)[0])
    return np.array(yh, dtype=np.float32)

def main(a):
    # load raw test + rul
    tr, te, test_rul = load_fd(f"data/CMAPSS/{a.fd}", a.fd)
    mu, sd = cache['norm']['mu'], cache['norm']['sd']
    units, X = last_windows_for_test(te, L=a.L)
    Xn = ((X - mu)/sd).astype(np.float32)

    # ground truth is RUL per unit in the same order
    y_true = test_rul.values.astype(np.float32)[:len(units)]
    os.makedirs("artifacts/eval", exist_ok=True)

    results = {}
    if a.model in ('lstm','both'):
        yhat = predict_lstm(cache, Xn, outdir=a.lstm_dir, hidden=a.lstm_h, layers=a.lstm_layers)
        m = summarise_metrics(y_true, yhat); results['lstm'] = m
        pd.DataFrame({"unit":units,"true":y_true,"pred":yhat}).to_csv("artifacts/eval/preds_lstm.csv", index=False)
    if a.model in ('gnn','both'):
        yhat = predict_gnn(cache, Xn, outdir=a.gnn_dir, hg=a.hg, ht=a.ht, layers=a.gnn_layers)
        m = summarise_metrics(y_true, yhat); results['gnn'] = m
        pd.DataFrame({"unit":units,"true":y_true,"pred":yhat}).to_csv("artifacts/eval/preds_gnn.csv", index=False)

    with open("artifacts/eval/test_metrics.json","w") as f: json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument('--fd', default='FD001')
    p.add_argument('--L', type=int, default=50)
    p.add_argument('--model', choices=['lstm','gnn','both'], default='both')
    p.add_argument('--lstm_dir', default='artifacts/lstm'); p.add_argument('--lstm_h', type=int, default=128); p.add_argument('--lstm_layers', type=int, default=2)
    p.add_argument('--gnn_dir', default='artifacts/gnn');  p.add_argument('--hg', type=int, default=64);       p.add_argument('--ht', type=int, default=128); p.add_argument('--gnn_layers', type=int, default=2)
    args=p.parse_args()
    cache=torch.load("data/cache/fd001_L50_s1_cache.pt", map_location='cpu', weights_only=False)
    main(args)
