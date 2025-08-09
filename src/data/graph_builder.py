import numpy as np, torch
from torch_geometric.utils import dense_to_sparse
def build_sensor_graph(df, sensor_cols, thresh=0.3, topk=None):
    X = df[sensor_cols].values.astype(np.float32)
    C = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(C, 0.0)
    if topk is not None:
        A = np.zeros_like(C)
        idx = np.argsort(-np.abs(C), axis=1)[:, :topk]
        rows = np.repeat(np.arange(C.shape[0])[:,None], topk, axis=1)
        A[rows, idx] = 1.0
        A = np.maximum(A, A.T)
    else:
        A = (np.abs(C) >= thresh).astype(np.float32)
    A = torch.from_numpy(A)
    edge_index, _ = dense_to_sparse(A)
    return edge_index
