import os
import torch
from src.config.settings import CKPT_ROOT


def save_checkpoint(model, optimizer, epoch, val_dice, filename="best_model.pth"):
    """Persist the model state, optimizer state, and metadata so training can
    be resumed or the best model can be loaded for inference later."""
    os.makedirs(CKPT_ROOT, exist_ok=True)
    path = os.path.join(CKPT_ROOT, filename)
    torch.save({
        "epoch":      epoch,
        "val_dice":   val_dice,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }, path)
    return path


def load_checkpoint(model, optimizer=None, filename="best_model.pth", device="cpu"):
    """Restore a previously saved checkpoint.  If `optimizer` is None (inference
    mode) the optimizer state is silently ignored.  Returns the epoch and
    val_dice that were recorded when the checkpoint was saved."""
    path = os.path.join(CKPT_ROOT, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    print(f"[Checkpoint] Loaded epoch {ckpt['epoch']}  |  val_dice {ckpt['val_dice']:.4f}")
    return ckpt["epoch"], ckpt["val_dice"]
