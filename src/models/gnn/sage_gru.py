import torch, torch.nn as nn
from torch_geometric.nn import SAGEConv
def batch_edge_index(edge_index, num_nodes, batch_size, device):
    offsets = (torch.arange(batch_size, device=device)*num_nodes).view(-1,1,1)
    ei = edge_index.t().unsqueeze(0).repeat(batch_size,1,1) + offsets
    return ei.reshape(-1,2).t().contiguous()
class SAGEGRU(nn.Module):
    def __init__(self, in_feats=1, hidden_g=64, hidden_t=128):
        super().__init__()
        self.s1 = SAGEConv(in_feats, hidden_g); self.s2 = SAGEConv(hidden_g, hidden_g)
        self.gru = nn.GRU(hidden_g, hidden_t, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_t,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x_seq, edge_index):
        B,T,N,F = x_seq.shape; dev = x_seq.device
        ei = batch_edge_index(edge_index.to(dev), N, B, dev)
        outs=[]
        for t in range(T):
            x = x_seq[:,t].reshape(B*N,F)
            h = torch.relu(self.s1(x, ei)); h = torch.relu(self.s2(h, ei))
            outs.append(h.view(B,N,-1).mean(1))
        H,_ = self.gru(torch.stack(outs,1))
        return self.head(H[:,-1,:]).squeeze(-1)
