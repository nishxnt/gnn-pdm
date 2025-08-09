import numpy as np, torch
from src.data.preprocess import load_fd, add_rul_labels, SENSOR_COLS
from src.data.graph_builder import build_sensor_graph
from src.models.gnn.sage_gru import SAGEGRU

if __name__ == "__main__":
    tr, te, _ = load_fd("data/CMAPSS/FD001", "FD001")
    tr = add_rul_labels(tr)
    edge_index = build_sensor_graph(tr, SENSOR_COLS, thresh=0.3)

    # Build one sample: last 50 cycles of the first unit
    unit = tr["unit"].unique()[0]
    dfu = tr[tr.unit==unit].tail(50)
    x_seq = dfu[SENSOR_COLS].values[:, :, None]  # [T,N,1]
    x_seq = np.expand_dims(x_seq, 0)             # [1,T,N,1]

    model = SAGEGRU(in_feats=1)
    with torch.no_grad():
        out = model(torch.tensor(x_seq, dtype=torch.float32), edge_index)
        print("Pred shape:", out.shape, "Value:", float(out))
