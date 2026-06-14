"""Entry point - full training and evaluation pipeline (PyTorch).

Supported datasets (--dataset): im2latex (printed, default) and
mathwriting (handwritten, deepcopy/MathWriting-human).
"""

import argparse
import os
import platform
import sys

import torch

from vit_latex.config import (DATASET_SIZE, TEST_SIZE, BLEU_NUM_SAMPLES,
                              MODEL_SAVE_PATH, CHECKPOINT_FILEPATH,
                              TRAINING_HISTORY_PLOT, PREDICTIONS_PLOT,
                              BEAM_WIDTH, DATASETS, DEFAULT_DATASET)
from vit_latex.data import (load_dataset, load_formulas, create_tokenizer,
                            prepare_training_data)
from vit_latex.training import (build_model, train_model, load_model,
                                plot_training_history, save_model)
from vit_latex.evaluation import (evaluate_model, visualize_predictions,
                                  batch_generate_captions, calculate_bleu_score,
                                  interactive_demo)


def setup_environment():
    """Print system information."""
    print(f"Python Platform: {platform.platform()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python {sys.version}")
    if torch.cuda.is_available():
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(f"GPU Devices: {names}")
    else:
        print("GPU Devices: [] (falling back to CPU)")
    print()


def artifact_path(base_path, dataset):
    """Per-dataset artifact path: insert a dataset suffix for non-default
    datasets so runs do not overwrite each other's checkpoints and figures."""
    if dataset == DEFAULT_DATASET:
        return base_path
    root, ext = os.path.splitext(base_path)
    return f"{root}_{dataset}{ext}"


def prepare_data(dataset):
    """Load and prepare the training data (train split)."""
    print(f"Loading training data ({dataset})...")

    train_images, train_latex_texts = load_dataset(
        DATASET_SIZE, split="train", dataset=dataset)

    tokenizer = create_tokenizer(train_latex_texts)

    train_images, train_sequences = prepare_training_data(
        train_images, train_latex_texts, tokenizer
    )

    print(f"Training image shape: {tuple(train_images.shape)}")
    print(f"Training sequence shape: {tuple(train_sequences.shape)}")
    print(f"Vocabulary size: {tokenizer.vocabulary_size()}")

    return train_images, train_sequences, tokenizer


def train_pipeline(train_images, train_sequences, tokenizer, dataset):
    """Training pipeline."""
    print("Starting training pipeline...")

    print("Building model...")
    model = build_model(tokenizer, train_sequences)

    # Pick one sample image for the per-epoch generation preview
    print("Training...")
    sample_image = train_images[0]
    history = train_model(
        model, train_images, train_sequences, sample_image,
        checkpoint_path=artifact_path(CHECKPOINT_FILEPATH, dataset))

    print("Plotting training history...")
    plot_training_history(history, path=artifact_path(TRAINING_HISTORY_PLOT, dataset))

    print("Saving model...")
    save_model(model, path=artifact_path(MODEL_SAVE_PATH, dataset))

    return model, history


def evaluate_pipeline(model, tokenizer, dataset, compare_beam=False):
    """Evaluation pipeline (on the independent test split).

    With compare_beam=True, BLEU is computed twice on the same subset
    (greedy vs beam search) for a decoding-strategy comparison.
    """
    print("Starting evaluation pipeline...")

    # Load the test split and encode it with the same tokenizer
    print(f"Loading test data ({dataset})...")
    test_images, test_latex_texts = load_dataset(
        TEST_SIZE, split="test", dataset=dataset)
    _, test_sequences = prepare_training_data(test_images, test_latex_texts, tokenizer)

    print("Evaluating model performance...")
    evaluate_model(model, test_images, test_sequences)

    # Autoregressive BLEU on a test subset (the headline generation metric;
    # teacher-forcing accuracy above overestimates real decoding quality)
    n_bleu = min(BLEU_NUM_SAMPLES, len(test_images))
    print(f"Computing BLEU on {n_bleu} test samples (greedy decoding)...")
    predictions = batch_generate_captions(model, test_images[:n_bleu])
    bleu = calculate_bleu_score(predictions, test_latex_texts[:n_bleu])
    if bleu is not None:
        print(f"BLEU score (greedy): {bleu:.4f}")

    if compare_beam:
        print(f"Computing BLEU on the same {n_bleu} samples "
              f"(beam search, width {BEAM_WIDTH})...")
        beam_predictions = batch_generate_captions(
            model, test_images[:n_bleu], beam_width=BEAM_WIDTH)
        beam_bleu = calculate_bleu_score(beam_predictions, test_latex_texts[:n_bleu])
        if beam_bleu is not None and bleu is not None:
            print()
            print("=== Decoding comparison ===")
            print(f"BLEU (greedy):           {bleu:.4f}")
            print(f"BLEU (beam, width {BEAM_WIDTH}):   {beam_bleu:.4f}")
            print(f"Delta:                   {beam_bleu - bleu:+.4f}")

    print("Generating sample predictions...")
    ground_truth = [t.replace('[START]', '').replace('[END]', '').strip()
                    for t in test_latex_texts]
    visualize_predictions(model, test_images[:5], ground_truth[:5],
                          path=artifact_path(PREDICTIONS_PLOT, dataset))

    # Interactive demo: only in an interactive terminal, so non-interactive
    # runs (e.g. remote training) do not block
    if sys.stdin.isatty():
        print("Starting interactive demo...")
        try:
            interactive_demo(model, test_images, ground_truth)
        except (KeyboardInterrupt, EOFError):
            print("Demo stopped")


def parse_args():
    parser = argparse.ArgumentParser(description="Train / evaluate the image-to-LaTeX model")
    parser.add_argument(
        '--dataset', choices=sorted(DATASETS), default=DEFAULT_DATASET,
        help=f"dataset to train/evaluate on (default: {DEFAULT_DATASET})")
    parser.add_argument(
        '--eval-only', action='store_true',
        help="skip training: rebuild the model and load weights from --weights")
    parser.add_argument(
        '--weights', default=None,
        help="weights file for --eval-only (default: the dataset's saved model)")
    parser.add_argument(
        '--beam', action='store_true',
        help="also compute BLEU with beam search for a decoding comparison")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    print("=== Vision Transformer Image-to-LaTeX Model (PyTorch) ===")

    setup_environment()

    if args.eval_only:
        # Only the train formulas are needed to rebuild the tokenizer
        # vocabulary (no image preprocessing); the TokenOutput prior bias is
        # restored from the saved state_dict anyway
        print(f"Loading training formulas ({args.dataset}, tokenizer fitting only)...")
        train_texts = load_formulas(DATASET_SIZE, split="train", dataset=args.dataset)
        tokenizer = create_tokenizer(train_texts)
        print(f"Vocabulary size: {tokenizer.vocabulary_size()}")

        weights = args.weights or artifact_path(MODEL_SAVE_PATH, args.dataset)
        print(f"Evaluation only: loading weights from {weights}")
        model = build_model(tokenizer, tokenizer(train_texts))
        load_model(model, weights)
    else:
        train_images, train_sequences, tokenizer = prepare_data(args.dataset)
        model, _ = train_pipeline(train_images, train_sequences, tokenizer, args.dataset)

    evaluate_pipeline(model, tokenizer, args.dataset, compare_beam=args.beam)

    print("=== Done ===")


if __name__ == "__main__":
    main()
