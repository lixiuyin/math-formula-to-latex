"""Tokenizer unit tests."""

import torch

from vit_latex.config import MAX_SEQ_LENGTH
from vit_latex.data import LatexTokenizer, PAD_ID, UNK_ID, PAD_TOKEN, UNK_TOKEN


def make_tokenizer(texts):
    tok = LatexTokenizer(max_tokens=600, max_seq_length=MAX_SEQ_LENGTH)
    tok.adapt(texts)
    return tok


def test_reserves_pad_and_unk_at_fixed_indices():
    # Arrange / Act
    tok = make_tokenizer(["[START] a b [END]"])
    vocab = tok.get_vocabulary()

    # Assert
    assert vocab[PAD_ID] == PAD_TOKEN
    assert vocab[UNK_ID] == UNK_TOKEN
    assert tok.word_to_index(PAD_TOKEN) == PAD_ID
    assert tok.word_to_index(UNK_TOKEN) == UNK_ID


def test_encode_pads_to_fixed_length_with_zeros():
    # Arrange
    tok = make_tokenizer(["[START] a b [END]"])

    # Act
    ids = tok.encode("[START] a b [END]")

    # Assert
    assert len(ids) == MAX_SEQ_LENGTH
    assert all(i == PAD_ID for i in ids[4:])


def test_encode_truncates_overlong_sequences():
    # Arrange
    tok = make_tokenizer(["[START] a [END]"])
    long_text = " ".join(["a"] * (MAX_SEQ_LENGTH + 50))

    # Act
    ids = tok.encode(long_text)

    # Assert
    assert len(ids) == MAX_SEQ_LENGTH


def test_decode_roundtrip_skips_padding():
    # Arrange
    tok = make_tokenizer(["[START] \\frac { a } { b } [END]"])
    text = "[START] \\frac { a } { b } [END]"

    # Act
    decoded = tok.decode(tok.encode(text))

    # Assert
    assert decoded == text


def test_maps_out_of_vocabulary_tokens_to_unk():
    # Arrange
    tok = make_tokenizer(["[START] a [END]"])

    # Act
    ids = tok.encode("never_seen_token")

    # Assert
    assert ids[0] == UNK_ID


def test_call_returns_long_tensor_batch():
    # Arrange
    tok = make_tokenizer(["[START] a b [END]"])

    # Act
    batch = tok(["[START] a [END]", "[START] b [END]"])

    # Assert
    assert batch.shape == (2, MAX_SEQ_LENGTH)
    assert batch.dtype == torch.long
