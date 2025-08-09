import argparse, os, numpy as np, torch, torch.nn as nn, mlflow
from torch.utils.data import DataLoader
from src.data.dataset import WindowDataset
from src.utils.seed import set_seed
from src.utils.metrics import summarise_metrics
from src.utils.checkpoint import save_checkpoint, load_if_exists
from src.models.gnn.sage_gru import SAGEGRU
def train_epoch(model, dl, opt, loss_fn, ei, dev):
    model.train(); total=0
    for x,y in dl:
        x=x.to(dev); y=y.to(dev)
        opt.zero_grad(); loss=loss_fn(model(x, ei),y); loss.backward(); opt.step()
        total+=loss.item()*len(y)
    return total/len(dl.dataset)
@torch.no_grad()
def eval_epoch(model, dl, ei, dev):
    model.eval(); ys=[]; yh=[]
    for x,y in dl:
        x=x.to(dev); ys.append(y.numpy()); yh.append(model(x, ei).cpu().numpy())
    y=np.concatenate(ys); yhat=np.concatenate(yh); return summarise_metrics(y,yhat)
def main(a):
    set_seed(a.seed); dev='cuda' if torch.cuda.is_available() else 'cpu'
    c=torch.load(a.cache,map_location='cpu', weights_only=False); X,y=c['X'],c['y']; tr,va=c['train_idx'],c['val_idx']; ei=c['edge_index'].to(dev)
    tr_dl=DataLoader(WindowDataset(X,y,tr), batch_size=a.batch, shuffle=True, num_workers=2)
    va_dl=DataLoader(WindowDataset(X,y,va), batch_size=a.batch, shuffle=False, num_workers=2)
    m=SAGEGRU(in_feats=1, hidden_g=a.hg, hidden_t=a.ht).to(dev)
    opt=torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4); loss_fn=nn.L1Loss()
    ck='artifacts/gnn'; os.makedirs(ck,exist_ok=True); last=f'{ck}/last.ckpt'; best=f'{ck}/best.ckpt'
    start,best_mae,_=load_if_exists(last, m, opt, None, map_location=dev)
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI','file:./mlruns')); mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME','gnn-pdm-fd001'))
    with mlflow.start_run(run_name='sage_gru'):
        mlflow.log_params({'hg':a.hg,'ht':a.ht,'lr':a.lr,'batch':a.batch,'epochs':a.epochs})
        for ep in range(start, a.epochs):
            tl=train_epoch(m,tr_dl,opt,loss_fn,ei,dev); metrics=eval_epoch(m,va_dl,ei,dev)
            save_checkpoint(last, m, opt, None, ep, best_mae, vars(a))
            if metrics['mae']<best_mae: best_mae=metrics['mae']; save_checkpoint(best, m, opt, None, ep, best_mae, vars(a))
            mlflow.log_metrics({'train_loss':tl, **metrics}, step=ep); print(f'Epoch {ep:03d}  train {tl:.4f}  val {metrics}')
        print('Best val MAE:', best_mae)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache', default='data/cache/fd001_L50_s1_cache.pt')
    p.add_argument('--epochs', type=int, default=30); p.add_argument('--batch', type=int, default=64)
    p.add_argument('--hg', type=int, default=64); p.add_argument('--ht', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--seed', type=int, default=42)
    main(p.parse_args())
