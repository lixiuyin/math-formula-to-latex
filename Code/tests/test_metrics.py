"""Masked loss / accuracy unit tests."""

import torch

from vit_latex.training import masked_loss, masked_acc


def test_masked_loss_ignores_padding_positions():
    # Arrange: perfect logits on real positions, garbage on the padded one
    vocab = 10
    preds = torch.full((1, 3, vocab), -10.0)
    preds[0, 0, 3] = 10.0
    preds[0, 1, 5] = 10.0
    labels = torch.tensor([[3, 5, 0]])  # last position is padding

    # Act
    loss = masked_loss(preds, labels)

    # Assert
    assert loss.item() < 0.01


def test_masked_loss_excludes_banned_token_labels():
    # Arrange: label points at a banned token whose logit is -1e9
    vocab = 10
    preds = torch.zeros(1, 2, vocab)
    preds[..., 1] = -1e9  # banned (e.g. [UNK])
    labels = torch.tensor([[3, 1]])

    # Act
    loss = masked_loss(preds, labels)

    # Assert: the ~1e9 per-token loss must not leak into the mean
    assert loss.item() < 10


def test_masked_acc_counts_only_non_padding_positions():
    # Arrange: correct on position 0, wrong on position 1, padding on 2
    vocab = 10
    preds = torch.full((1, 3, vocab), -10.0)
    preds[0, 0, 3] = 10.0
    preds[0, 1, 9] = 10.0
    preds[0, 2, 9] = 10.0
    labels = torch.tensor([[3, 5, 0]])

    # Act
    acc = masked_acc(preds, labels)

    # Assert: 1 correct out of 2 real positions
    assert abs(acc.item() - 0.5) < 1e-6
