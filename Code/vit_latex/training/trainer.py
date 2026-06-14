"""Training loop: validation split, early stopping, best-weights checkpointing."""

import math
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from vit_latex.config import (BATCH_SIZE, EPOCHS, VALIDATION_SPLIT,
                              CHECKPOINT_FILEPATH, MODEL_SAVE_PATH,
                              EMBEDDING_DIM, MAX_SEQ_LEN_1, DROPOUT_RATE,
                              DECODER_LAYERS, DECODER_HEADS, IMG_SHAPE,
                              EARLY_STOPPING_PATIENCE, TRAINING_HISTORY_PLOT,
                              GRAD_CLIP_NORM, SPLIT_SEED)
from vit_latex.models import vision_transformer_encoder, Captioner, TokenOutput
from vit_latex.plotting import finalize_figure
from vit_latex.training.metrics import masked_loss, masked_acc
from vit_latex.training.scheduler import create_optimizer_and_scheduler


def get_device():
    """Use GPU when available, otherwise fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _save_weights(model, filepath):
    """Save a state_dict, creating the parent directory if needed."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(model.state_dict(), filepath)


def build_model(tokenizer, train_sequences):
    """Build the full model."""
    # Vision Transformer encoder
    feature_extractor = vision_transformer_encoder(IMG_SHAPE)

    # Output layer: fit the prior bias on the label distribution
    # (sequences[:, 1:], matching the training targets, without [START]).
    # [START] has count 0 -> bias -1e9, so it is never generated at inference.
    output_layer = TokenOutput(tokenizer, banned_tokens=('', '[UNK]'))
    output_layer.adapt(train_sequences[:, 1:])

    model = Captioner(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        output_layer=output_layer,
        units=EMBEDDING_DIM,
        dropout_rate=DROPOUT_RATE,
        num_layers=DECODER_LAYERS,
        num_heads=DECODER_HEADS,
        max_length=MAX_SEQ_LEN_1,
    )

    return model


def _run_epoch(model, loader, device, optimizer=None, scheduler=None):
    """Run one epoch; validation mode when optimizer is None.

    Returns (mean loss, mean accuracy).
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    with torch.set_grad_enabled(is_train):
        for images, sequences in loader:
            images = images.to(device, non_blocking=True)
            sequences = sequences.to(device, non_blocking=True)
            inputs, labels = sequences[:, :-1], sequences[:, 1:]

            preds = model(images, inputs)
            loss = masked_loss(preds, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            total_acc += masked_acc(preds.detach(), labels).item()
            n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def train_model(model, train_images, train_sequences, sample_image=None,
                checkpoint_path=CHECKPOINT_FILEPATH):
    """Train the model: validation split, early stopping, best-weights
    checkpointing, and a sample generation after each epoch.

    The train/val split is random with a fixed seed (SPLIT_SEED): the dataset
    may be ordered (e.g. by formula length), so a trailing slice would give a
    biased validation set. Returns a history dict with the same keys as a
    Keras History: loss / val_loss / masked_acc / val_masked_acc.
    """
    device = get_device()
    model.to(device)
    print(f"Training device: {device}")

    n_total = train_images.shape[0]
    n_val = int(n_total * VALIDATION_SPLIT)
    n_train = n_total - n_val

    # Subset over a seeded permutation: reproducible split, no data copy
    generator = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(n_total, generator=generator).tolist()
    dataset = TensorDataset(train_images, train_sequences)
    train_ds = Subset(dataset, perm[:n_train])
    val_ds = Subset(dataset, perm[n_train:])
    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin)

    optimizer, scheduler = create_optimizer_and_scheduler(model)

    history = {'loss': [], 'val_loss': [], 'masked_acc': [], 'val_masked_acc': []}
    best_val_loss = math.inf
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, device, optimizer, scheduler)
        val_loss, val_acc = _run_epoch(model, val_loader, device)

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['masked_acc'].append(train_acc)
        history['val_masked_acc'].append(val_acc)

        print(f"Epoch {epoch}/{EPOCHS} - "
              f"loss: {train_loss:.4f} - masked_acc: {train_acc:.4f} - "
              f"val_loss: {val_loss:.4f} - val_masked_acc: {val_acc:.4f}")

        # Generate one sample per epoch (equivalent of the Keras GenerateText callback)
        if sample_image is not None:
            print()
            print(model.simple_gen(sample_image, temperature=0))
            print()

        # Best-weights checkpointing + early stopping
        # (equivalent of ModelCheckpoint / EarlyStopping)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            _save_weights(model, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Validation loss has not improved for "
                      f"{EARLY_STOPPING_PATIENCE} epochs; stopping early")
                break

    # Restore the best weights (equivalent of restore_best_weights=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history


def plot_training_history(history, path=TRAINING_HISTORY_PLOT):
    """Plot the training history."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Transformer model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(history['masked_acc'])
    plt.plot(history['val_masked_acc'])
    plt.title('Transformer model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')

    finalize_figure(path)


def save_model(model, path=MODEL_SAVE_PATH):
    """Save model weights (state_dict, including the TokenOutput prior bias buffer).

    To reload, rebuild the architecture with build_model(...) and then call
    load_model(model, path).
    """
    _save_weights(model, path)
    print(f"Model saved to: {path}")


def load_model(model, model_path):
    """Load weights into an already-built model instance."""
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model
