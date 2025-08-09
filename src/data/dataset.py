import torch
from torch.utils.data import Dataset
class WindowDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X, self.y, self.idx = X, y, indices
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return torch.tensor(self.X[j], dtype=torch.float32), torch.tensor(self.y[j], dtype=torch.float32)
