import pandas as pd
from pathlib import Path
SENSOR_COLS = [f"s{i:02d}" for i in range(1, 22)]
OP_COLS = ["op1","op2","op3"]
ALL_COLS = ["unit","cycle"] + OP_COLS + SENSOR_COLS

def load_fd(root: str, fd: str="FD001"):
    root = Path(root)
    tr = pd.read_csv(root/f"train_{fd}.txt", sep=r"\s+", header=None)
    te = pd.read_csv(root/f"test_{fd}.txt",  sep=r"\s+", header=None)
    rul = pd.read_csv(root/f"RUL_{fd}.txt",   sep=r"\s+", header=None)[0]
    tr.columns = te.columns = ALL_COLS
    return tr, te, rul

def add_rul_labels(train: pd.DataFrame, cap: int=130) -> pd.DataFrame:
    max_cycle = train.groupby("unit")["cycle"].transform("max")
    rul = (max_cycle - train["cycle"]).clip(upper=cap)
    return train.assign(RUL=rul)
