import argparse, os, numpy as np, torch, mlflow, xgboost as xgb
from src.utils.metrics import summarise_metrics
def make_features(X):  # X: [B,T,N,1]
    X = X[...,0]; B,T,N = X.shape; feats=[]
    for b in range(B):
        x=X[b]; mu=x.mean(0); sd=x.std(0); mn=x.min(0); mx=x.max(0)
        t=np.arange(T,dtype=np.float32); t=(t-t.mean())/(t.std()+1e-8)
        sl=((x*t[:,None]).sum(0)-x.sum(0)*t.sum()/T)/((t**2).sum()-(t.sum()**2)/T+1e-8)
        feats.append(np.concatenate([mu,sd,mn,mx,sl],0))
    return np.stack(feats,0)
def main(args):
    cache=torch.load(args.cache, map_location='cpu', weights_only=False)
    X, y = cache['X'], cache['y']; tr,va = cache['train_idx'], cache['val_idx']
    Ftr, Fva = make_features(X[tr]), make_features(X[va])
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI','file:./mlruns'))
    mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME','gnn-pdm-fd001'))
    with mlflow.start_run(run_name='xgb_baseline'):
        dtr=xgb.DMatrix(Ftr,label=y[tr]); dva=xgb.DMatrix(Fva,label=y[va])
        params={'max_depth':6,'eta':0.05,'subsample':0.9,'colsample_bytree':0.9,'objective':'reg:squarederror'}
        model=xgb.train(params,dtr,num_boost_round=800,evals=[(dva,'val')],verbose_eval=False)
        yhat=model.predict(dva); m=summarise_metrics(y[va], yhat)
        for k,v in m.items(): mlflow.log_metric(k,float(v))
        os.makedirs('artifacts/xgb', exist_ok=True); model.save_model('artifacts/xgb/model.json'); mlflow.log_artifact('artifacts/xgb/model.json')
        print('XGB metrics:', m)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--cache', default='data/cache/fd001_L50_s1_cache.pt'); main(p.parse_args())
