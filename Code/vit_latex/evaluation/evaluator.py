"""Model evaluation and inference (PyTorch)."""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from vit_latex.config import (BATCH_SIZE, PREDICTIONS_PLOT,
                              DEMO_RANDOM_SAMPLES, DEMO_DIR)
from vit_latex.data import preprocess_image
from vit_latex.plotting import finalize_figure
from vit_latex.training import masked_loss, masked_acc, get_device

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}


def evaluate_model(model, test_images, test_sequences):
    """Evaluate model performance (masked loss / masked acc, padding ignored)."""
    device = get_device()
    model.to(device)
    model.eval()

    loader = DataLoader(
        TensorDataset(test_images, test_sequences), batch_size=BATCH_SIZE)

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for images, sequences in loader:
            images = images.to(device)
            sequences = sequences.to(device)
            inputs, labels = sequences[:, :-1], sequences[:, 1:]

            preds = model(images, inputs)
            total_loss += masked_loss(preds, labels).item()
            total_acc += masked_acc(preds, labels).item()
            n_batches += 1

    test_loss = total_loss / max(n_batches, 1)
    test_accuracy = total_acc / max(n_batches, 1)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    return test_loss, test_accuracy


def generate_caption(model, image, temperature=0, beam_width=None):
    """Generate a caption for a single image.

    With beam_width=None, uses greedy / temperature sampling (simple_gen);
    with a beam_width (e.g. 4), uses beam search decoding (beam_gen).
    """
    if isinstance(image, str):
        # Input is a file path: load and preprocess the image
        image = preprocess_image(Image.open(image))

    if beam_width:
        return model.beam_gen(image, beam_width=beam_width)
    return model.simple_gen(image, temperature=temperature)


def batch_generate_captions(model, images, temperature=0, beam_width=None):
    """Generate captions for multiple images."""
    captions = []
    for image in images:
        caption = generate_caption(model, image, temperature, beam_width=beam_width)
        captions.append(caption)
    return captions


def visualize_predictions(model, images, true_captions=None, num_samples=5,
                          path=PREDICTIONS_PLOT):
    """Visualize prediction results."""
    if len(images) > num_samples:
        indices = np.random.choice(len(images), num_samples, replace=False)
        images = [images[i] for i in indices]
        if true_captions:
            true_captions = [true_captions[i] for i in indices]

    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    if len(images) == 1:
        axes = [axes]

    for i, image in enumerate(images):
        predicted_caption = generate_caption(model, image)

        # Show the image (squeeze the channel dim: (1, H, W) -> (H, W))
        axes[i].imshow(np.squeeze(image.cpu().numpy()), cmap='gray')
        axes[i].axis('off')

        title = f"Predicted: {predicted_caption}"
        if true_captions:
            title += f"\nGround truth: {true_captions[i]}"
        axes[i].set_title(title, fontsize=10)

    finalize_figure(path)


def calculate_bleu_score(predictions, references):
    """Compute the BLEU-4 score (requires nltk).

    LaTeX formulas are space-separated and case-sensitive (e.g. \\Gamma vs
    \\gamma), so tokens are split on whitespace without lowercasing, and the
    [START]/[END] special tokens are removed. Uses method1 smoothing so short
    sequences or missing n-gram overlaps do not collapse the score to 0.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        print("nltk is required to compute the BLEU score")
        return None

    smoothing = SmoothingFunction().method1
    special_tokens = {'[START]', '[END]'}
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = [t for t in pred.split() if t not in special_tokens]
        ref_tokens = [t for t in ref.split() if t not in special_tokens]
        score = sentence_bleu([ref_tokens], pred_tokens,
                              smoothing_function=smoothing)
        bleu_scores.append(score)

    return np.mean(bleu_scores)


def evaluate_on_test_set(model, test_data, tokenizer):
    """Evaluate the model on the test set."""
    test_images, test_captions = test_data

    predictions = batch_generate_captions(model, test_images)

    # BLEU score (None when nltk is not installed)
    bleu_score = calculate_bleu_score(predictions, test_captions)

    if bleu_score is not None:
        print(f"BLEU score: {bleu_score:.4f}")

    visualize_predictions(model, test_images[:5], test_captions[:5])

    return predictions, bleu_score


def _demo_single_image(model, image_path, save_figure=True):
    """Caption one image file and optionally save an annotated figure."""
    caption = generate_caption(model, image_path)
    print(f"{image_path}: {caption}")

    if save_figure:
        image = preprocess_image(Image.open(image_path))
        plt.figure(figsize=(6, 4))
        plt.imshow(np.squeeze(image.numpy()), cmap='gray')
        plt.title(f"Predicted: {caption}", fontsize=9)
        plt.axis('off')
        stem = os.path.splitext(os.path.basename(image_path))[0]
        finalize_figure(os.path.join(DEMO_DIR, f"demo_{stem}.png"))


def _list_image_files(directory):
    """Image files directly inside a directory, sorted by name."""
    entries = sorted(os.listdir(directory))
    return [os.path.join(directory, name) for name in entries
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS]


def _demo_random_samples(model, images, captions=None, num_samples=DEMO_RANDOM_SAMPLES):
    """Caption randomly sampled images and save annotated figures."""
    indices = np.random.choice(len(images), size=min(num_samples, len(images)),
                               replace=False)
    print(f"Sampling {len(indices)} random test image(s)...")
    for idx in indices:
        idx = int(idx)
        caption = generate_caption(model, images[idx])
        print(f"sample {idx}: {caption}")

        plt.figure(figsize=(6, 4))
        plt.imshow(np.squeeze(images[idx].cpu().numpy()), cmap='gray')
        title = f"Predicted: {caption}"
        if captions is not None:
            title += f"\nGround truth: {captions[idx]}"
        plt.title(title, fontsize=9)
        plt.axis('off')
        finalize_figure(os.path.join(DEMO_DIR, f"demo_random_{idx}.png"))


def interactive_demo(model, images=None, captions=None):
    """Interactive demo.

    images/captions: optional pool (e.g. the test split) sampled on empty input.
    """
    print("Interactive image captioning demo")
    print("Enter an image file or a directory of images; 'quit' to exit")
    if images is not None:
        print("Press Enter without a path to caption random test images")

    while True:
        image_path = input("Image path: ").strip()

        if image_path.lower() == 'quit':
            break

        try:
            if not image_path:
                if images is None:
                    print("No test images available; please enter a file path")
                    continue
                _demo_random_samples(model, images, captions)
            elif os.path.isdir(image_path):
                files = _list_image_files(image_path)
                if not files:
                    print(f"No image files found in directory: {image_path}")
                    continue
                print(f"Found {len(files)} image(s) in {image_path}")
                for path in files:
                    _demo_single_image(model, path)
            else:
                _demo_single_image(model, image_path)

        except Exception as e:
            print(f"Error: {e}")
            print("Please check that the image path is correct")


def save_predictions(predictions, filepath):
    """Save prediction results to a file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, pred in enumerate(predictions):
            f.write(f"Image_{i}: {pred}\n")
    print(f"Predictions saved to: {filepath}")


def load_predictions(filepath):
    """Load prediction results from a file."""
    predictions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                pred = line.split(':', 1)[1].strip()
                predictions.append(pred)
    return predictions
