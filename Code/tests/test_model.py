"""End-to-end model tests: forward, backward, generation, save/load."""

import torch

from vit_latex.config import MAX_SEQ_LENGTH, MAX_SEQ_LEN_1
from vit_latex.training import (build_model, load_model, masked_loss,
                                create_optimizer_and_scheduler)


def test_forward_produces_expected_logit_shape(model, images, sequences):
    # Act
    preds = model(images, sequences[:, :-1])

    # Assert
    assert preds.shape == (4, MAX_SEQ_LENGTH - 1, model.tokenizer.vocabulary_size())


def test_decode_truncates_sequences_beyond_position_capacity(model, images, sequences):
    # Arrange: a sequence longer than the position embedding capacity
    overlong = torch.cat([sequences, sequences], dim=1)  # length 304 > 151

    # Act: must not raise an index error
    preds = model(images, overlong)

    # Assert
    assert preds.shape[1] == MAX_SEQ_LEN_1


def test_training_step_decreases_nothing_but_runs(model, images, sequences):
    # Arrange
    opt, sched = create_optimizer_and_scheduler(model)

    # Act
    preds = model(images, sequences[:, :-1])
    loss = masked_loss(preds, sequences[:, 1:])
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if sched is not None:
        sched.step()

    # Assert
    assert torch.isfinite(loss)


def test_simple_gen_returns_string_without_special_tokens(model, images):
    # Act
    result = model.simple_gen(images[0], temperature=0)

    # Assert
    assert isinstance(result, str)
    assert '[START]' not in result
    assert '[END]' not in result


def test_beam_gen_returns_string_without_special_tokens(model, images):
    # Act
    result = model.beam_gen(images[0], beam_width=2)

    # Assert
    assert isinstance(result, str)
    assert '[START]' not in result
    assert '[END]' not in result


def test_generation_restores_training_mode(model, images):
    # Arrange
    model.train()

    # Act
    model.simple_gen(images[0])

    # Assert
    assert model.training


def test_state_dict_roundtrip_preserves_prior_bias(model, tokenizer, sequences, tmp_path):
    # Arrange
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    rebuilt = build_model(tokenizer, sequences)

    # Act
    load_model(rebuilt, str(path))

    # Assert
    assert torch.equal(rebuilt.output_layer.bias, model.output_layer.bias)
