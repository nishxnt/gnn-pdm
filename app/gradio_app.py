import gradio as gr, torch, numpy as np, pandas as pd
from src.models.gnn.sage_gru import SAGEGRU

CACHE = "data/cache/fd001_L50_s1_cache.pt"
CKPT  = "artifacts/gnn/best.ckpt"

def predict(csv):
    df = pd.read_csv(csv.name)
    cache = torch.load(CACHE, map_location='cpu', weights_only=False)
    mu, sd = cache["norm"]["mu"], cache["norm"]["sd"]
    need = [f"s{i:02d}" for i in range(1,22)]
    assert all(c in df.columns for c in need), "CSV must include s01..s21 columns"
    x = df.tail(50)[need].values.astype(np.float32)[...,None]     # [L,N,1]
    x = ((x - mu)/sd)[None,...]
    m = SAGEGRU(in_feats=1, hidden_g=64, hidden_t=128)
    m.load_state_dict(torch.load(CKPT, map_location='cpu')["model"]); m.eval()
    y = m(torch.tensor(x), cache["edge_index"]).item()
    return float(y)

demo = gr.Interface(fn=predict, inputs=gr.File(file_types=['.csv']), outputs="number",
                    title="GNN RUL Predictor (FD001)", allow_flagging="never")
if __name__ == "__main__":
    demo.launch()
