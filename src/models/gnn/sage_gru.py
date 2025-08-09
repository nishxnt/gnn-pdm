import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class SAGEGRU(nn.Module):
    """
    Minimal SAGE+GRU for Sprint 0 smoke test.
    NOTE: forward assumes batch size = 1 for now.
    """
    def __init__(self, in_feats=1, hidden_g=64, hidden_t=128, num_layers=1):
        super().__init__()
        self.sage1 = SAGEConv(in_feats, hidden_g)
        self.sage2 = SAGEConv(hidden_g, hidden_g)
        self.gru = nn.GRU(hidden_g, hidden_t, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_t, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x_seq, edge_index):
        # x_seq: [1, T, N, F]
        B, T, N, F = x_seq.shape
        assert B == 1, "Sprint 0 smoke test expects batch size 1"
        outs = []
        for t in range(T):
            x = x_seq[:, t].reshape(N, F)  # [N,F]
            h = self.sage1(x, edge_index)
            h = torch.relu(h)
            h = self.sage2(h, edge_index)
            h = torch.relu(h)
            h = h.mean(dim=0, keepdim=True)  # [1, hidden_g]
            outs.append(h)
        H = torch.stack(outs, dim=1).squeeze(0)  # [T, hidden_g]
        H,_ = self.gru(H.unsqueeze(0))           # [1,T,hidden_t]
        y = self.head(H[:, -1, :]).squeeze(-1)   # [1]
        return y
