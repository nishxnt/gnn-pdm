import torch, torch.nn as nn, inspect
from torch_geometric.nn import SAGEConv

class SAGEGRU(nn.Module):
    def __init__(self, in_feats=1, hidden_g=64, hidden_t=128, layers=2, dropout=0.1):
        super().__init__()
        self.hidden_g = hidden_g
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

    def _sage_once(self, x, edge_index, edge_weight, conv, ln):
        # Robustly support PyG variants that lack 'edge_weight' in SAGEConv.forward
        if edge_weight is not None and 'edge_weight' in inspect.signature(conv.forward).parameters:
            x = conv(x, edge_index, edge_weight=edge_weight)   # supported build
        else:
            x = conv(x, edge_index)                            # older build
        x = torch.relu(ln(x))
        x = self.drop(x)
        return x

    def _sage_stack(self, x, edge_index, edge_weight=None):
        # x: [N, F]
        for conv, ln in zip(self.sages, self.norms):
            x = self._sage_once(x, edge_index, edge_weight, conv, ln)
        return x                                                # [N, hidden_g]

    def forward(self, x_seq, edge_index, edge_weight=None):
        """
        x_seq: [B, T, N, F]
        edge_index: [2, E] single graph (shared across batch)
        """
        B, T, N, F = x_seq.shape
        device = x_seq.device

        edge_index = edge_index.to(device=device, dtype=torch.long)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device=device, dtype=torch.float32)

        # Per-sample, per-time graph pass (CPU/GPU-safe & simple)
        H_list = []
        for b in range(B):
            Hb = []
            for t in range(T):
                xt = x_seq[b, t].reshape(N, F)                 # [N, F]
                h = self._sage_stack(xt, edge_index, edge_weight)  # [N, hidden_g]
                h = h.mean(dim=0)                              # [hidden_g]
                Hb.append(h)
            H_list.append(torch.stack(Hb, dim=0))              # [T, hidden_g]
        H = torch.stack(H_list, dim=0)                         # [B, T, hidden_g]

        _, ht = self.gru(H)                                    # [1, B, hidden_t]
        y = self.head(ht[-1]).squeeze(-1)                      # [B]
        return y
