import argparse, os, csv, numpy as np, torch, torch.nn as nn, mlflow
from torch.utils.data import DataLoader
from src.data.dataset import WindowDataset
from src.models.lstm_reg import LSTMReg
from src.utils.metrics import summarise_metrics
from src.utils.seed import set_seed
from src.utils.checkpoint import save_checkpoint, load_if_exists

def train_one_epoch(model, dl, opt, loss_fn, device):
    model.train(); total=0.0
    for x,y in dl:
        x=x.squeeze(-1).to(device); y=y.to(device)
        opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step()
        total += float(loss.item())*len(y)
    return total/len(dl.dataset)

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval(); ys=[]; yh=[]
    for x,y in dl:
        x=x.squeeze(-1).to(device)
        ys.append(y.numpy()); yh.append(model(x).cpu().numpy())
    y=np.concatenate(ys); yhat=np.concatenate(yh)
    return summarise_metrics(y,yhat)

def log_history(row, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["epoch","train_loss","mae","rmse","phm"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow([row[k] for k in header])

def main(a):
    set_seed(a.seed); dev='cuda' if torch.cuda.is_available() else 'cpu'
    c=torch.load(a.cache,map_location='cpu', weights_only=False)
    X,y=c['X'],c['y']; tr,va=c['train_idx'],c['val_idx']
    tr_dl=DataLoader(WindowDataset(X,y,tr), batch_size=a.batch, shuffle=True, num_workers=2)
    va_dl=DataLoader(WindowDataset(X,y,va), batch_size=a.batch, shuffle=False, num_workers=2)
    m=LSTMReg(n_in=X.shape[2], hidden=a.hidden, layers=a.layers, dropout=a.dropout).to(dev)
    opt=torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4); loss_fn=nn.L1Loss()
    ck='artifacts/lstm'; os.makedirs(ck,exist_ok=True)
    last=f'{ck}/last.ckpt'; best=f'{ck}/best.ckpt'
    start,best_mae,_=load_if_exists(last, m, opt, None, map_location=dev)

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI','file:./mlruns'))
    mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME','gnn-pdm-fd001'))
    with mlflow.start_run(run_name='lstm_baseline'):
        mlflow.log_params(vars(a))
        for ep in range(start, a.epochs):
            tl=train_one_epoch(m,tr_dl,opt,loss_fn,dev)
            metrics=evaluate(m,va_dl,dev)
            save_checkpoint(last, m, opt, None, ep, best_mae, vars(a))
            if metrics['mae']<best_mae:
                best_mae=metrics['mae']; save_checkpoint(best, m, opt, None, ep, best_mae, vars(a))
            
            mlflow.log_metric('train_loss', float(tl), step=ep)
            for k in ('mae','rmse','phm'):
                mlflow.log_metric(k, float(metrics[k]), step=ep)
            log_history({"epoch":ep,"train_loss":float(tl), **{k:float(metrics[k]) for k in ('mae','rmse','phm')}}, f"{ck}/history.csv")
            print(f"Epoch {ep:03d}  train {tl:.4f}  val {metrics}")
        print('Best val MAE:', best_mae)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache', default='data/cache/fd001_L50_s1_cache.pt')
    p.add_argument('--epochs', type=int, default=20); p.add_argument('--batch', type=int, default=64)
    p.add_argument('--hidden', type=int, default=128); p.add_argument('--layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.1); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    main(p.parse_args())
