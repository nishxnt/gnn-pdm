import numpy as np
def mae(y, yhat): return float(np.mean(np.abs(yhat - y)))
def rmse(y, yhat): return float(np.sqrt(np.mean((yhat - y)**2)))
def phm_score(y, yhat):
    d = yhat - y
    s = np.where(d < 0, np.exp(-d/13) - 1, np.exp(d/10) - 1)
    return float(np.sum(s))
