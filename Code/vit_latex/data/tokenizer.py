"""Whitespace LaTeX tokenizer with a frequency-ordered vocabulary."""

import collections
import re

import torch

from vit_latex.config import MAX_SEQ_LENGTH, VOCAB_SIZE

PAD_TOKEN = ''
UNK_TOKEN = '[UNK]'
PAD_ID = 0
UNK_ID = 1

# Raw LaTeX token pattern: commands (\frac, \operatorname*), escaped symbols
# (\{, \%), or any single non-space character (digits split individually,
# matching the im2latex convention).
LATEX_TOKEN_PATTERN = re.compile(r'\\[a-zA-Z]+\*?|\\.|\S')


def tokenize_latex(formula):
    """Split a raw (non-spaced) LaTeX string into space-separated tokens.

    Example: 'V(\\tilde{\\beta})' -> 'V ( \\tilde { \\beta } )'.
    Applying it to an already space-separated formula is a no-op in terms of
    the resulting token sequence.
    """
    return ' '.join(LATEX_TOKEN_PATTERN.findall(formula))


class LatexTokenizer:
    """Whitespace tokenizer for LaTeX (case- and symbol-preserving, e.g. { } ^ _ \\).

    Vocabulary convention (matching Keras TextVectorization):
      index 0 -> '' (padding), index 1 -> '[UNK]', remaining tokens ordered by
      descending frequency, capped at max_tokens; encoded sequences are padded
      or truncated to max_seq_length.
    """

    def __init__(self, max_tokens=VOCAB_SIZE, max_seq_length=MAX_SEQ_LENGTH):
        self.max_tokens = max_tokens
        self.max_seq_length = max_seq_length
        self._vocab = [PAD_TOKEN, UNK_TOKEN]
        self._token_to_id = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}

    def adapt(self, texts):
        counts = collections.Counter()
        for text in texts:
            counts.update(text.split())

        most_common = counts.most_common(self.max_tokens - 2)
        self._vocab = [PAD_TOKEN, UNK_TOKEN] + [tok for tok, _ in most_common]
        self._token_to_id = {tok: i for i, tok in enumerate(self._vocab)}

    def vocabulary_size(self):
        return len(self._vocab)

    def get_vocabulary(self):
        return list(self._vocab)

    def word_to_index(self, token):
        return self._token_to_id.get(token, UNK_ID)

    def index_to_word(self, index):
        if 0 <= index < len(self._vocab):
            return self._vocab[index]
        return UNK_TOKEN

    def encode(self, text):
        """Single text -> fixed-length id list (truncated / zero-padded)."""
        ids = [self.word_to_index(tok) for tok in text.split()]
        ids = ids[:self.max_seq_length]
        ids += [PAD_ID] * (self.max_seq_length - len(ids))
        return ids

    def decode(self, ids):
        """Id sequence -> space-joined token string (padding skipped)."""
        tokens = [self.index_to_word(int(i)) for i in ids if int(i) != PAD_ID]
        return ' '.join(tokens)

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return torch.tensor([self.encode(t) for t in texts], dtype=torch.long)


def create_tokenizer(latex_texts, vocab_size=VOCAB_SIZE, max_seq_length=MAX_SEQ_LENGTH):
    """Create and fit a tokenizer."""
    tokenizer = LatexTokenizer(max_tokens=vocab_size, max_seq_length=max_seq_length)
    tokenizer.adapt(latex_texts)
    return tokenizer
