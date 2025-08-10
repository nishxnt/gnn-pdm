import itertools, subprocess, csv, os, json, time, argparse, pathlib

CFG = [
    {"L":50, "hg":64,  "ht":128, "lr":1e-3, "layers":2},
    {"L":50, "hg":128, "ht":256, "lr":5e-4, "layers":2},
    {"L":30, "hg":64,  "ht":128, "lr":1e-3, "layers":2},
]
OUT = pathlib.Path("artifacts/sweeps"); OUT.mkdir(parents=True, exist_ok=True)

def run_one(i,c):
    run = f"sweep_gnn_{i}"
    outdir = f"artifacts/gnn_{i}"
    cmd = ["python","-m","src.train_gnn","--cache","data/cache/fd001_L50_s1_cache.pt",
           "--epochs","20","--batch","64","--hg",str(c["hg"]), "--ht",str(c["ht"]),
           "--lr",str(c["lr"]), "--layers",str(c["layers"]), "--outdir",outdir,
           "--run_name", run, "--amp"]
    print(">>", " ".join(cmd)); subprocess.check_call(cmd)
    # evaluate GNN on test
    subprocess.check_call(["python","-m","src.eval","--model","gnn","--gnn_dir",outdir])
    with open("artifacts/eval/test_metrics.json") as f: metrics=json.load(f)["gnn"]
    row = {"run":run, **c, **metrics}
    return row

if __name__=="__main__":
    rows=[]
    for i,c in enumerate(CFG,1):
        rows.append(run_one(i,c))
    with open(OUT/"summary.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("Saved", OUT/"summary.csv")
