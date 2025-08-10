import os, torch

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_mae, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch + 1,
        "best_mae": best_mae,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "config": config,
    }, path)

def load_if_exists(path, model, optimizer=None, scheduler=None,
                   map_location="cpu", strict=False):
    """
    Returns (start_epoch, best_mae, saved_config) or (0, inf, None) if missing/incompatible.
    """
    if not os.path.exists(path):
        return 0, float("inf"), None

    sd = torch.load(path, map_location=map_location, weights_only=False)
    start = int(sd.get("epoch", 0))
    best = float(sd.get("best_mae", float("inf")))
    try:
        model.load_state_dict(sd["model"], strict=strict)
    except Exception as e:
        print("⚠️  Incompatible checkpoint; ignoring and starting fresh:", e)
        return 0, float("inf"), None

    if optimizer is not None and sd.get("optimizer") is not None:
        optimizer.load_state_dict(sd["optimizer"])
    if scheduler is not None and sd.get("scheduler") is not None:
        scheduler.load_state_dict(sd["scheduler"])

    return start, best, sd.get("config")
