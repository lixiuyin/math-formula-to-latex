"""Transformer decoder building blocks (PyTorch)."""

import numpy as np
import torch
import torch.nn as nn

from vit_latex.config import EMBEDDING_DIM, DROPOUT_RATE
from vit_latex.data.tokenizer import PAD_ID


class SeqEmbedding(nn.Module):
    """Sequence embedding: token embedding + learnable position embedding."""

    def __init__(self, vocab_size, max_length, depth):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, depth, padding_idx=PAD_ID)
        self.pos_embedding = nn.Embedding(max_length, depth)

    def forward(self, seq):
        positions = torch.arange(seq.shape[1], device=seq.device)
        return self.token_embedding(seq) + self.pos_embedding(positions)


class CausalSelfAttention(nn.Module):
    """Causal self-attention (residual + LayerNorm)."""

    def __init__(self, num_heads, embed_dim, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        attn, _ = self.mha(x, x, x, attn_mask=causal_mask, need_weights=False)
        return self.layernorm(x + attn)


class CrossAttention(nn.Module):
    """Cross-attention (query = text, key/value = image features)."""

    def __init__(self, num_heads, embed_dim, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.last_attention_scores = None

    def forward(self, x, y):
        attn, attention_scores = self.mha(x, y, y, need_weights=True)
        self.last_attention_scores = attention_scores
        return self.layernorm(x + attn)


class FeedForward(nn.Module):
    """Feed-forward network (residual + LayerNorm)."""

    def __init__(self, units, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(units, 2 * units),
            nn.ReLU(),
            nn.Linear(2 * units, units),
            nn.Dropout(dropout_rate),
        )
        self.layernorm = nn.LayerNorm(units)

    def forward(self, x):
        return self.layernorm(x + self.seq(x))


class DecoderLayer(nn.Module):
    """Decoder layer: causal self-attention -> cross-attention -> feed-forward."""

    def __init__(self, units, num_heads=1, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.self_attention = CausalSelfAttention(
            num_heads=num_heads, embed_dim=units, dropout=dropout_rate)
        self.cross_attention = CrossAttention(
            num_heads=num_heads, embed_dim=units, dropout=dropout_rate)
        self.ff = FeedForward(units=units, dropout_rate=dropout_rate)

    def forward(self, in_seq, out_seq):
        out_seq = self.self_attention(out_seq)
        out_seq = self.cross_attention(out_seq, in_seq)
        self.last_attention_scores = self.cross_attention.last_attention_scores
        return self.ff(out_seq)


class TokenOutput(nn.Module):
    """Token output layer: linear projection to vocabulary + frequency prior bias
    (banned tokens set to -1e9).

    The bias is registered as a buffer, so it is saved and loaded with the
    state_dict. Outputs logits (no softmax), consistent with cross_entropy /
    argmax / multinomial(softmax(logits/T)) / log_softmax usage.
    """

    def __init__(self, tokenizer, banned_tokens=('', '[UNK]')):
        super().__init__()
        vocab_size = tokenizer.vocabulary_size()
        self.dense = nn.Linear(EMBEDDING_DIM, vocab_size)
        self.tokenizer = tokenizer
        self.banned_tokens = banned_tokens
        self.register_buffer('bias', torch.zeros(vocab_size))

    def adapt(self, sequences):
        """Initialize the output bias from the token distribution of training labels.

        sequences: (N, T) LongTensor of training targets (without [START]).
        """
        vocab_size = self.tokenizer.vocabulary_size()
        counts_arr = torch.bincount(
            sequences.reshape(-1), minlength=vocab_size).double().numpy()

        for token in self.banned_tokens:
            counts_arr[self.tokenizer.word_to_index(token)] = 0

        total = counts_arr.sum()
        p = counts_arr / total
        p[counts_arr == 0] = 1.0
        log_p = np.log(p)  # log(1) == 0

        entropy = -(log_p * p).sum()
        print()
        print(f"Uniform entropy: {np.log(vocab_size):0.2f}")
        print(f"Marginal entropy: {entropy:0.2f}")

        log_p[counts_arr == 0] = -1e9
        with torch.no_grad():
            self.bias.copy_(torch.from_numpy(log_p).float())

    def forward(self, x):
        return self.dense(x) + self.bias
