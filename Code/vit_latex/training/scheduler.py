"""Optimizer and learning rate schedule."""

import torch

from vit_latex.config import (EMBEDDING_DIM, LEARNING_RATE, WEIGHT_DECAY,
                              USE_WARMUP_SCHEDULE, WARMUP_STEPS)


def transformer_lr_lambda(step, d_model=EMBEDDING_DIM, warmup_steps=WARMUP_STEPS):
    """Warmup learning rate from the original Transformer paper
    (Vaswani et al., 2017, Sec. 5.3).

    lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})

    The learning rate increases linearly during the first warmup_steps steps
    (avoiding divergence from unstable second-moment estimates early in
    training), then decays with the inverse square root of the step number.
    Use with LambdaLR and base_lr=1.0.
    """
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)


def create_optimizer_and_scheduler(model):
    """Create the optimizer and LR scheduler.

    Defaults to the original Transformer warmup schedule (with the paper's
    beta_2=0.98 and epsilon=1e-9); falls back to a constant learning rate
    when USE_WARMUP_SCHEDULE=False.
    """
    if USE_WARMUP_SCHEDULE:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1.0, weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.98), eps=1e-9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, transformer_lr_lambda)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = None
    return optimizer, scheduler
