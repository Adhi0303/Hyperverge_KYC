import torch

def dice_loss(logits, target, eps=1e-6):
    """Differentiable (soft) Dice loss = 1 - Dice, computed per sample on the
    sigmoid probabilities and averaged over the batch. It is combined with BCE
    during training to pair a pixel-wise term with an overlap term."""
    prob = torch.sigmoid(logits)                          # logits -> probabilities in [0, 1]
    numerator = 2 * (prob * target).sum(dim=(1, 2, 3)) + eps       # soft intersection, per sample
    denominator = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps  # soft area sum
    dice_per_sample = numerator / denominator
    return (1 - dice_per_sample).mean()
