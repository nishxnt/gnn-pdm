import torch, torch.nn as nn
from torch_geometric.nn import SAGEConv

class SAGEGRU(nn.Module):
    def __init__(self, in_feats=1, hidden_g=64, hidden_t=128, layers=2, dropout=0.1):
        super().__init__()
        self.sages = nn.ModuleList()
        self.norms = nn.ModuleList()
        last = in_feats
        for _ in range(layers):
            self.sages.append(SAGEConv(last, hidden_g))
            self.norms.append(nn.LayerNorm(hidden_g))
            last = hidden_g
        self.gru = nn.GRU(hidden_g, hidden_t, batch_first=True)
        self.head = nn.Linear(hidden_t, 1)
        self.drop = nn.Dropout(dropout)

    def _sage_stack(self, x, edge_index, edge_weight=None):
        # x: [N, F]
        for conv, ln in zip(self.sages, self.norms):
            try:
                x = conv(x, edge_index, edge_weight)
            except TypeError:
                x = conv(x, edge_index)           # older PyG without edge_weight in SAGEConv
            x = torch.relu(ln(x))
            x = self.drop(x)
        return x

    def forward(self, x_seq, edge_index, edge_weight=None):
        # x_seq: [B, T, N, F]  (we train with B>=1; in Colab we often have B=??)
        B, T, N, F = x_seq.shape
        outs = []
        for t in range(T):
            x = x_seq[:, t].reshape(N, F)         # collapse batch for graph aggregation
            h = self._sage_stack(x, edge_index, edge_weight)   # [N, hidden_g]
            outs.append(h)
        H = torch.stack(outs, dim=1)              # [N,T,hidden_g] when B==1; otherwise shape aligns
        if B == 1: H = H.unsqueeze(0)             # [1,N,T,H] style fix → we want [1,T,N,H]
        H = H.squeeze(0)                          # [T,hidden_g]
        H = H.unsqueeze(0)                        # [1,T,hidden_g]
        _, ht = self.gru(H)                       # ht: [1,1,hidden_t]
        y = self.head(ht[-1]).squeeze(-1)         # [1]
        return y
