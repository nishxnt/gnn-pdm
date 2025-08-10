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
        # x: [nodes_total, F]
        for conv, ln in zip(self.sages, self.norms):
            try:
                x = conv(x, edge_index, edge_weight)
            except TypeError:
                x = conv(x, edge_index)  # older PyG
            x = torch.relu(ln(x))
            x = self.drop(x)
        return x

    def forward(self, x_seq, edge_index, edge_weight=None):
        """
        x_seq: [B, T, N, F]
        edge_index: [2, E] for one graph of N nodes (shared across batch)
        """
        B, T, N, F = x_seq.shape
        device = x_seq.device

        # Build a block-diagonal batched graph: repeat edges with offsets of N
        # shape: [2, E*B]
        offsets = (torch.arange(B, device=device).repeat_interleave(edge_index.size(1)) * N)
        ei_b = edge_index.repeat(1, B) + offsets
        ew_b = edge_weight.repeat(B) if edge_weight is not None else None

        # For each time step: run GNN on [B*N, F], then mean over nodes per sample
        H_bt = []
        for t in range(T):
            xt = x_seq[:, t].reshape(B * N, F)        # [B*N, F]
            h = self._sage_stack(xt, ei_b, ew_b)      # [B*N, hidden_g]
            h = h.view(B, N, -1).mean(dim=1)          # [B, hidden_g]
            H_bt.append(h)

        H = torch.stack(H_bt, dim=1)                  # [B, T, hidden_g]
        _, ht = self.gru(H)                           # ht: [1, B, hidden_t]
        y = self.head(ht[-1]).squeeze(-1)             # [B]
        return y
