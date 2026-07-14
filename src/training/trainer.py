import torch
import torch.nn as nn
from tqdm import tqdm

from src.training.losses import dice_loss
from src.evaluation.metrics import dice_coeff


def run_epoch(model, loader, optimizer, device, is_train=True):
    """Run one pass through `loader`.

    Training mode  : computes combined BCE + Dice loss, back-propagates, and
                     updates the model weights.
    Validation mode: runs the forward pass only (torch.no_grad) so no gradients
                     are stored, saving memory.

    Returns
    -------
    avg_loss : float   mean combined loss over the epoch
    avg_dice : float   mean pixel Dice coefficient over the epoch
    """
    bce = nn.BCEWithLogitsLoss()            # stable numerically — takes raw logits
    model.train(is_train)

    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, masks in tqdm(loader, desc="train" if is_train else "val", leave=False):
            images = images.to(device, non_blocking=True)   # non_blocking = async GPU transfer
            masks  = masks.to(device, non_blocking=True)

            logits = model(images)                          # (N, 1, H, W) raw logits

            # -- Combined loss: BCE gives a per-pixel signal;
            #    Dice drives the overlap metric directly. --
            loss = 0.5 * bce(logits, masks) + 0.5 * dice_loss(logits, masks)

            if is_train:
                optimizer.zero_grad(set_to_none=True)       # slightly faster than zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # -- Accumulate metrics (detach from computation graph) --
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                batch_dice = dice_coeff(
                    preds.cpu().numpy(),
                    masks.cpu().numpy(),
                )

            total_loss += loss.item()
            total_dice += batch_dice
            n_batches  += 1

    return total_loss / n_batches, total_dice / n_batches
