import os, torch
def save_checkpoint(path, model, optimizer, scheduler, epoch, best, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch, "best": best, "config": config,
    }, tmp); os.replace(tmp, path)

def load_if_exists(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    if not os.path.exists(path): return 0, float("inf"), None
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and ckpt["optimizer"]: optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt["scheduler"]: scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch",0)+1, ckpt.get("best", float("inf")), ckpt.get("config")
