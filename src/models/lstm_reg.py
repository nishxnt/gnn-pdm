import torch.nn as nn
class LSTMReg(nn.Module):
    def __init__(self, n_in, hidden=128, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, dropout=dropout if layers>1 else 0.0, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self, x):  # [B,T,N]
        h,_ = self.lstm(x); return self.head(h[:,-1,:]).squeeze(-1)
