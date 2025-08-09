import os, pathlib, pandas as pd, matplotlib.pyplot as plt, mlflow
from mlflow.tracking import MlflowClient

PLOTS_DIR = pathlib.Path("artifacts/plots"); PLOTS_DIR.mkdir(parents=True, exist_ok=True)
mlruns_root = os.getenv("MLFLOW_TRACKING_URI","file:./mlruns").replace("file:","")
exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME","gnn-pdm-fd001")
mlflow.set_tracking_uri(f"file:{mlruns_root}")
exp = mlflow.get_experiment_by_name(exp_name)
assert exp, f"Experiment '{exp_name}' not found."
client = MlflowClient()

def latest_run_id(run_name):
    df = mlflow.search_runs([exp.experiment_id],
                            filter_string=f"tags.mlflow.runName = '{run_name}'",
                            order_by=["start_time DESC"])
    return None if df.empty else df.iloc[0]["run_id"]

def plot_series(epochs, vals, title, key, tag):
    plt.figure(figsize=(6,4))
    plt.plot(epochs, vals, marker="o")
    plt.title(title); plt.xlabel("epoch"); plt.ylabel(key); plt.tight_layout()
    out = PLOTS_DIR / f"{tag}_{key}.png"; plt.savefig(out); plt.close(); print("saved:", out)

def plot_run(run_name, csv_path, tag):
    rid = latest_run_id(run_name)
    keys = ("train_loss","mae","rmse","phm")
    any_ok = False
    if rid:
        for key in keys:
            hist = client.get_metric_history(rid, key)
            if hist:
                epochs = [m.step for m in hist]; vals = [m.value for m in hist]
                plot_series(epochs, vals, f"{run_name} — {key} (mlflow)", key, tag)
                any_ok = True
    if (not any_ok) and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for key in [k for k in keys if k in df.columns]:
            plot_series(df["epoch"], df[key], f"{run_name} — {key} (history.csv)", key, tag)
        any_ok = True
    if not any_ok:
        print(f"⚠️ No metrics found for {run_name}. Run training first.")

if __name__ == "__main__":
    plot_run("lstm_baseline", "artifacts/lstm/history.csv", "lstm")
    plot_run("sage_gru", "artifacts/gnn/history.csv", "gnn")
