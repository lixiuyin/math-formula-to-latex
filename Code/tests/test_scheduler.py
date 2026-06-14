"""Learning rate schedule unit tests."""

from vit_latex.config import EMBEDDING_DIM, WARMUP_STEPS
from vit_latex.training import transformer_lr_lambda


def test_warmup_increases_linearly_before_peak():
    # Arrange / Act
    quarter = transformer_lr_lambda(WARMUP_STEPS // 4)
    half = transformer_lr_lambda(WARMUP_STEPS // 2)

    # Assert
    assert half > quarter
    assert abs(half / quarter - 2.0) < 0.01


def test_lr_peaks_at_warmup_steps_then_decays():
    # Arrange / Act
    peak = transformer_lr_lambda(WARMUP_STEPS)
    before = transformer_lr_lambda(WARMUP_STEPS - 100)
    after = transformer_lr_lambda(WARMUP_STEPS * 4)

    # Assert
    assert peak > before
    assert peak > after
    expected_peak = EMBEDDING_DIM ** -0.5 * WARMUP_STEPS ** -0.5
    assert abs(peak - expected_peak) < 1e-9


def test_step_zero_does_not_divide_by_zero():
    # Act / Assert
    assert transformer_lr_lambda(0) > 0
