import argparse, os, csv, numpy as np, torch, torch.nn as nn, mlflow
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from src.data.dataset import WindowDataset
from src.utils.seed import set_seed
from src.utils.metrics import summarise_metrics
from src.utils.checkpoint import save_checkpoint, load_if_exists
from src.models.gnn.sage_gru import SAGEGRU
from src.utils.early_stop import EarlyStopper

def _make_outdir(base): os.makedirs(base, exist_ok=True); return base
def _log_hist(row, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdr = ["epoch","train_loss","mae","rmse","phm"]
    write_header = not os.path.exists(path)
    with open(path,"a",newline="") as f:
        w=csv.writer(f); 
        if write_header: w.writerow(hdr)
        w.writerow([row.get(k) for k in hdr])

def train_epoch(model, dl, opt, loss_fn, ei, dev, scaler, use_amp, clip=1.0):
    model.train(); total=0.0
    for x,y in dl:
        x=x.to(dev); y=y.to(dev)
        opt.zero_grad()
        with autocast(enabled=use_amp):
            loss = loss_fn(model(x, ei), y)
        scaler.scale(loss).backward()
        if clip: nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt); scaler.update()
        total += float(loss.item())*len(y)
    return total/len(dl.dataset)

@torch.no_grad()
def eval_epoch(model, dl, ei, dev):
    model.eval(); ys=[]; yh=[]
    for x,y in dl:
        x=x.to(dev)
        ys.append(y.numpy()); yh.append(model(x, ei).cpu().numpy())
    y=np.concatenate(ys); yhat=np.concatenate(yh)
    return summarise_metrics(y,yhat)

def main(a):
    set_seed(a.seed)
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    c=torch.load(a.cache,map_location='cpu', weights_only=False)
    X,y=c['X'],c['y']; tr,va=c['train_idx'],c['val_idx']
    ei, ew = (c['edge_index'].to(dev), None)
    # edge weights optional in cache; if missing, ignore
    if 'edge_weight' in c: ew = c['edge_weight'].to(dev)

    tr_dl=DataLoader(WindowDataset(X,y,tr), batch_size=a.batch, shuffle=True, num_workers=2)
    va_dl=DataLoader(WindowDataset(X,y,va), batch_size=a.batch, shuffle=False, num_workers=2)

    m=SAGEGRU(in_feats=1, hidden_g=a.hg, hidden_t=a.ht, layers=a.layers, dropout=a.dropout).to(dev)
    opt=torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    loss_fn=nn.L1Loss()
    out=_make_outdir(a.outdir)
    last, best = f"{out}/last.ckpt", f"{out}/best.ckpt"
    start,best_mae,_=load_if_exists(last, m, opt, None, map_location=dev)
    hist=f"{out}/history.csv"
    stopper=EarlyStopper(patience=a.patience, min_delta=a.min_delta)
    scaler=GradScaler(enabled=a.amp)

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI','file:./mlruns'))
    mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME','gnn-pdm-fd001'))
    with mlflow.start_run(run_name=a.run_name):
        mlflow.log_params({'hg':a.hg,'ht':a.ht,'layers':a.layers,'dropout':a.dropout,
                           'lr':a.lr,'batch':a.batch,'epochs':a.epochs,'wd':a.wd,
                           'patience':a.patience,'min_delta':a.min_delta,'amp':a.amp,
                           'outdir':a.outdir})
        if start >= a.epochs:
            metrics = eval_epoch(m, va_dl, (ei,ew), dev)
            mlflow.log_metric('train_loss', float('nan'), step=start)
            for k in ('mae','rmse','phm'): mlflow.log_metric(k, float(metrics[k]), step=start)
            _log_hist({"epoch":start,"train_loss":float('nan'), **metrics}, hist)
            print(f"[resume] nothing to train (start={start} >= epochs={a.epochs}). Logged eval-only metrics:", metrics)
            print('Best val MAE:', best_mae); return

        for ep in range(start, a.epochs):
            tl = train_epoch(m, tr_dl, opt, loss_fn, (ei,ew), dev, scaler, a.amp, a.clip)
            metrics = eval_epoch(m, va_dl, (ei,ew), dev)
            save_checkpoint(last, m, opt, None, ep, best_mae, vars(a))
            if metrics['mae'] < best_mae:
                best_mae = metrics['mae']; save_checkpoint(best, m, opt, None, ep, best_mae, vars(a))
            mlflow.log_metric('train_loss', float(tl), step=ep)
            for k in ('mae','rmse','phm'): mlflow.log_metric(k, float(metrics[k]), step=ep)
            _log_hist({"epoch":ep,"train_loss":float(tl), **metrics}, hist)
            sch.step(metrics['mae'])
            print(f"Epoch {ep:03d}  train {tl:.4f}  val {metrics}")
            if stopper.step(metrics['mae']):
                print(f"Early stopping at epoch {ep}; best MAE={best_mae:.4f}")
                break
        print('Best val MAE:', best_mae)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache', default='data/cache/fd001_L50_s1_cache.pt')
    p.add_argument('--epochs', type=int, default=30); p.add_argument('--batch', type=int, default=64)
    p.add_argument('--hg', type=int, default=64); p.add_argument('--ht', type=int, default=128)
    p.add_argument('--layers', type=int, default=2); p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--amp', action='store_true'); p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=5); p.add_argument('--min_delta', type=float, default=0.0)
    p.add_argument('--outdir', default='artifacts/gnn'); p.add_argument('--run_name', default='sage_gru_v2')
    p.add_argument('--seed', type=int, default=42)
    a=p.parse_args(); main(a)
