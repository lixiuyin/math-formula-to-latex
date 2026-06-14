"""Raw-LaTeX tokenization tests (MathWriting-style unspaced formulas)."""

from vit_latex.config import DATASETS, DEFAULT_DATASET
from vit_latex.data import tokenize_latex


def test_splits_commands_braces_and_characters():
    # Arrange / Act
    tokens = tokenize_latex('V(\\tilde{\\beta})')

    # Assert
    assert tokens == 'V ( \\tilde { \\beta } )'


def test_keeps_starred_commands_as_one_token():
    # Act
    tokens = tokenize_latex('\\operatorname*{lim}_{x\\to0}')

    # Assert
    assert tokens.split()[0] == '\\operatorname*'
    assert '\\to' in tokens.split()


def test_splits_digits_individually_matching_im2latex_convention():
    # Act
    tokens = tokenize_latex('x^{12}')

    # Assert
    assert tokens == 'x ^ { 1 2 }'


def test_escaped_symbols_are_single_tokens():
    # Act
    tokens = tokenize_latex('50\\%+\\{a\\}')

    # Assert
    assert '\\%' in tokens.split()
    assert '\\{' in tokens.split()
    assert '\\}' in tokens.split()


def test_idempotent_on_already_spaced_formulas():
    # Arrange: an im2latex-style pre-tokenized formula
    spaced = '\\frac { a } { b } + x ^ { 2 }'

    # Act / Assert
    assert tokenize_latex(spaced) == spaced


def test_dataset_registry_is_consistent():
    # Assert: every dataset entry declares the fields the loader relies on
    assert DEFAULT_DATASET in DATASETS
    for cfg in DATASETS.values():
        assert {'name', 'formula_field', 'splits', 'pretokenized'} <= set(cfg)
        assert {'train', 'test'} <= set(cfg['splits'])
