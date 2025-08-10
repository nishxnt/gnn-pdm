import numpy as np, torch
from torch_geometric.utils import dense_to_sparse

def _topk_adjacency(C: np.ndarray, k: int) -> np.ndarray:
    n = C.shape[0]
    A = np.zeros_like(C, dtype=np.float32)
    idx = np.argsort(-np.abs(C), axis=1)[:, :k]
    rows = np.repeat(np.arange(n)[:, None], k, axis=1)
    A[rows, idx] = 1.0
    A = np.maximum(A, A.T)
    return A

def build_sensor_graph(df, sensor_cols, thresh: float = 0.3, topk: int | None = None,
                       return_weights: bool = True):
    """Returns edge_index and optional edge_weight (|corr|) for sensors."""
    X = df[sensor_cols].values.astype(np.float32)
    C = np.corrcoef(X, rowvar=False)  # [N,N]
    np.fill_diagonal(C, 0.0)

    if topk is not None:
        A = _topk_adjacency(C, topk)
    else:
        A = (np.abs(C) >= thresh).astype(np.float32)

    edge_index, _ = dense_to_sparse(torch.from_numpy(A))
    if not return_weights:
        return edge_index

    # edge weights = |corr| on existing edges
    W = np.abs(C) * A
    ew = []
    ei = edge_index.numpy()
    for i in range(ei.shape[1]):
        u, v = ei[:, i]
        ew.append(W[u, v])
    edge_weight = torch.tensor(ew, dtype=torch.float32)
    return edge_index, edge_weight
