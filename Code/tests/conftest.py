"""Shared fixtures: a tiny tokenizer/model pair fast enough for CPU tests."""

import matplotlib
import pytest
import torch

matplotlib.use('Agg')  # headless backend for any plotting under test

from vit_latex.config import MAX_SEQ_LENGTH, IMG_SHAPE
from vit_latex.data import LatexTokenizer
from vit_latex.training import build_model

SAMPLE_TEXTS = [
    "[START] \\frac { a } { b } + x ^ 2 [END]",
    "[START] \\sum _ { i = 1 } ^ n i [END]",
    "[START] a + b = c [END]",
    "[START] \\alpha \\beta \\gamma [END]",
]


@pytest.fixture(scope="session")
def tokenizer():
    tok = LatexTokenizer(max_tokens=600, max_seq_length=MAX_SEQ_LENGTH)
    tok.adapt(SAMPLE_TEXTS)
    return tok


@pytest.fixture(scope="session")
def sequences(tokenizer):
    return tokenizer(SAMPLE_TEXTS)


@pytest.fixture(scope="session")
def model(tokenizer, sequences):
    torch.manual_seed(0)
    return build_model(tokenizer, sequences)


@pytest.fixture(scope="session")
def images():
    torch.manual_seed(0)
    return torch.rand(4, 1, IMG_SHAPE[0], IMG_SHAPE[1])
