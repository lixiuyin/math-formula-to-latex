"""Masked loss and accuracy (padding positions excluded)."""

import torch.nn.functional as F

from vit_latex.data.tokenizer import PAD_ID


def masked_loss(preds, labels):
    """Cross-entropy that ignores padding positions.

    Positions whose per-token loss exceeds 1e8 are also masked out: these are
    labels for banned tokens (e.g. [UNK] from out-of-vocabulary test tokens),
    whose -1e9 output bias would otherwise blow up the mean loss.

    preds: (B, T, V) logits; labels: (B, T) LongTensor.
    """
    labels = labels.reshape(-1)
    loss = F.cross_entropy(
        preds.reshape(-1, preds.shape[-1]), labels, reduction='none')
    mask = ((labels != PAD_ID) & (loss < 1e8)).float()
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def masked_acc(preds, labels):
    """Accuracy over non-padding positions only."""
    mask = labels != PAD_ID
    match = (preds.argmax(dim=-1) == labels) & mask
    return match.sum().float() / mask.sum().clamp(min=1).float()
