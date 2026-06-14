"""Image-to-LaTeX captioner model (ViT encoder + Transformer decoder)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_latex.config import (EMBEDDING_DIM, MAX_SEQ_LEN_1, DROPOUT_RATE,
                              BEAM_WIDTH, LENGTH_PENALTY_ALPHA)
from vit_latex.models.decoder import SeqEmbedding, DecoderLayer


class Captioner(nn.Module):
    """Image-to-LaTeX sequence generation model."""

    def __init__(self, tokenizer, feature_extractor, output_layer, num_layers=1,
                 units=EMBEDDING_DIM, max_length=MAX_SEQ_LEN_1, num_heads=1,
                 dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.seq_embedding = SeqEmbedding(
            vocab_size=tokenizer.vocabulary_size(),
            depth=units,
            max_length=max_length)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(units, num_heads=num_heads, dropout_rate=dropout_rate)
            for _ in range(num_layers)])

        self.output_layer = output_layer

    @property
    def device(self):
        return next(self.parameters()).device

    def word_to_index(self, token):
        return self.tokenizer.word_to_index(token)

    def _decode(self, features, txt):
        """Compute output logits from image features and token ids
        (feature extraction not included)."""
        # Truncate to the maximum length supported by the position embedding
        # to avoid index overflow; guards every decoding path (forward,
        # simple_gen, beam_gen)
        txt = txt[:, :self.max_length]
        txt = self.seq_embedding(txt)
        for dec_layer in self.decoder_layers:
            txt = dec_layer(features, txt)
        return self.output_layer(txt)

    def forward(self, image, txt):
        """Forward pass: image (B, 1, H, W), txt (B, T) LongTensor -> (B, T, vocab)."""
        features = self.feature_extractor(image)
        return self._decode(features, txt)

    @torch.no_grad()
    def simple_gen(self, image, temperature=0):
        """Simple text generation (greedy / temperature sampling).

        image: a single image tensor of shape (1, H, W).
        """
        was_training = self.training
        self.eval()
        try:
            end_id = self.word_to_index('[END]')
            tokens = torch.tensor(
                [[self.word_to_index('[START]')]], dtype=torch.long, device=self.device)
            image = image.to(self.device).unsqueeze(0)  # (1, 1, H, W)
            features = self.feature_extractor(image)

            for _ in range(MAX_SEQ_LEN_1 - 1):
                preds = self._decode(features, tokens)[:, -1, :]  # (1, vocab)

                if temperature == 0:
                    next_token = preds.argmax(dim=-1, keepdim=True)  # (1, 1)
                else:
                    probs = F.softmax(preds / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

                tokens = torch.cat([tokens, next_token], dim=1)

                if int(next_token[0, 0]) == end_id:
                    break

            # Drop [START]; drop the last token only if it really is [END],
            # so a real token is not lost when max length is reached
            # before [END] is generated
            output_tokens = tokens[0, 1:].tolist()
            if output_tokens and output_tokens[-1] == end_id:
                output_tokens = output_tokens[:-1]

            return self.tokenizer.decode(output_tokens)
        finally:
            self.train(was_training)

    @torch.no_grad()
    def beam_gen(self, image, beam_width=BEAM_WIDTH, alpha=LENGTH_PENALTY_ALPHA):
        """Beam search decoding (Vaswani et al., 2017, Sec. 6.1:
        beam width 4, length penalty alpha=0.6).

        Keeps beam_width candidate sequences; at each step all active candidates
        are batched into a single forward pass, and the top beam_width of the
        "candidates x top-k expansions" pool are kept. Candidates that produce
        [END] move to the finished set. The best sequence is chosen by cumulative
        log-probability normalized with the GNMT length penalty
        lp(Y) = ((5+|Y|)/6)^alpha.
        """
        was_training = self.training
        self.eval()
        try:
            beam_width = max(1, int(beam_width))
            start_id = self.word_to_index('[START]')
            end_id = self.word_to_index('[END]')

            # Extract image features once, shared by all candidates
            image = image.to(self.device).unsqueeze(0)  # (1, 1, H, W)
            features = self.feature_extractor(image)    # (1, patches, depth)

            active = [(0.0, [start_id])]  # (cumulative log-prob, token sequence)
            finished = []

            for _ in range(MAX_SEQ_LEN_1 - 1):
                # All active candidates share the same length at a given step,
                # so they can be batched directly
                seqs = torch.tensor(
                    [tokens for _, tokens in active], dtype=torch.long, device=self.device)
                feats = features.expand(len(active), -1, -1)
                logits = self._decode(feats, seqs)[:, -1, :]  # (n_active, vocab)
                log_probs = F.log_softmax(logits, dim=-1).cpu().numpy()

                # Expand candidates x top-k, then keep the top beam_width overall
                candidates = []
                for i, (logp, tokens) in enumerate(active):
                    top_ids = np.argsort(log_probs[i])[-beam_width:]
                    candidates.extend(
                        (logp + float(log_probs[i][token_id]), tokens + [int(token_id)])
                        for token_id in top_ids)
                candidates.sort(key=lambda c: c[0], reverse=True)

                active = []
                for logp, tokens in candidates[:beam_width]:
                    if tokens[-1] == end_id:
                        finished.append((logp, tokens))
                    else:
                        active.append((logp, tokens))

                if not active or len(finished) >= beam_width:
                    break

            def normalized_score(logp, tokens):
                # GNMT length penalty over generated content tokens
                # (neither [START] nor a trailing [END] is counted)
                length = len(tokens) - 1 - (1 if tokens[-1] == end_id else 0)
                length_penalty = ((5.0 + length) / 6.0) ** alpha
                return logp / length_penalty

            pool = finished if finished else active
            _, best_tokens = max(pool, key=lambda c: normalized_score(c[0], c[1]))

            best_tokens = best_tokens[1:]  # drop [START]
            if best_tokens and best_tokens[-1] == end_id:
                best_tokens = best_tokens[:-1]  # drop [END]
            if not best_tokens:
                return ""

            return self.tokenizer.decode(best_tokens)
        finally:
            self.train(was_training)
