class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best = float('inf'); self.wait = 0

    def step(self, value):
        if value < self.best - self.min_delta:
            self.best = value; self.wait = 0; return False  # not stopping
        self.wait += 1
        return self.wait >= self.patience
