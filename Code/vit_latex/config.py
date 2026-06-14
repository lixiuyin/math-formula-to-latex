"""Configuration - all hyperparameters and settings (PyTorch implementation)."""

import os

# Dataset configuration (Hugging Face datasets)
# Each entry: HF name, formula column, split-name mapping, and whether the
# formulas are already space-separated tokens (im2latex) or raw LaTeX strings
# that need tokenization (MathWriting).
DATASETS = {
    'im2latex': {
        'name': "yuntian-deng/im2latex-100k",
        'formula_field': 'formula',
        'splits': {'train': 'train', 'test': 'test'},
        'pretokenized': True,
    },
    'mathwriting': {
        'name': "deepcopy/MathWriting-human",
        'formula_field': 'latex',
        'splits': {'train': 'train', 'test': 'test'},
        'pretokenized': False,
    },
}
DEFAULT_DATASET = 'im2latex'

DATASET_SIZE = 50000   # number of training samples (im2latex train: 55000;
                       # mathwriting train: ~230000 — capped for comparability)
TEST_SIZE = 2000       # number of evaluation samples

# Image processing configuration
IMG_SHAPE = [50, 200, 1]   # model input size (height, width, channels); raw images
                           # (RGB, varying sizes) are converted to grayscale and resized
PATCH_SIZE = 10

# Text / vocabulary configuration
VOCAB_SIZE = 600          # vocabulary cap (im2latex has ~500+ tokens),
                          # including padding ('') and [UNK]
MAX_SEQ_LENGTH = 152      # token sequence length (incl. [START]/[END];
                          # longer sequences truncated, shorter ones padded)
MAX_SEQ_LEN_1 = MAX_SEQ_LENGTH - 1

# Model configuration
EMBEDDING_DIM = 256

# Encoder (Vision Transformer) configuration
NUM_HEADS = 4
TRANSFORMER_UNITS = [
    EMBEDDING_DIM * 2,
    EMBEDDING_DIM,
]
TRANSFORMER_LAYERS = 8

# Decoder (Transformer decoder) configuration
DECODER_LAYERS = 4
DECODER_HEADS = 8

# Training configuration
BATCH_SIZE = 768   # ~9-11 GiB on a 32 GiB GPU; try 512 if memory allows
EPOCHS = 100
LEARNING_RATE = 1e-4   # constant learning rate (only used when USE_WARMUP_SCHEDULE=False)
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.1     # shared by encoder/decoder attention and FFN; 0.1 is the
                       # common choice for ViT and comparable im2latex models
VALIDATION_SPLIT = 0.2
SPLIT_SEED = 42        # seed for the random train/val split: the dataset may be
                       # ordered (e.g. by formula length), so a trailing slice
                       # would give a biased validation set
EARLY_STOPPING_PATIENCE = 10
GRAD_CLIP_NORM = 1.0   # max gradient norm; prevents a single bad batch near the
                       # warmup LR peak from blowing up the weights (observed:
                       # divergence at epoch 7 collapsing output to '{'/'}')

# Learning rate schedule (Vaswani et al., 2017 Sec. 5.3:
# linear warmup followed by inverse square-root decay)
USE_WARMUP_SCHEDULE = True
WARMUP_STEPS = 800   # peak LR = d_model^-0.5 * warmup^-0.5 ~= 2.2e-3; 500 gave
                      # ~2.8e-3, which diverged once near the peak at batch 768

# Inference configuration (Vaswani et al., 2017 Sec. 6.1:
# beam width 4, GNMT length penalty alpha=0.6)
BEAM_WIDTH = 4
LENGTH_PENALTY_ALPHA = 0.6

# Evaluation configuration
BLEU_NUM_SAMPLES = 500   # test samples used for autoregressive BLEU
                         # (greedy decoding is sequential, so full-split BLEU is slow)
DEMO_RANDOM_SAMPLES = 20  # test images sampled when the demo gets empty input

# Output path configuration: all artifacts live under outputs/
# (directories are created automatically before saving)
OUTPUT_DIR = 'outputs'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
DEMO_DIR = os.path.join(OUTPUT_DIR, 'demo')

CHECKPOINT_FILEPATH = os.path.join(
    CHECKPOINT_DIR, 'transformer_model_best.pt')  # best validation weights during training
MODEL_SAVE_PATH = os.path.join(
    CHECKPOINT_DIR, 'transformer_model.pt')       # final weights saved after training
TRAINING_HISTORY_PLOT = os.path.join(FIGURES_DIR, 'training_history.png')
PREDICTIONS_PLOT = os.path.join(FIGURES_DIR, 'predictions.png')

# Number of patches
NUM_PATCHES = (IMG_SHAPE[0] // PATCH_SIZE) * (IMG_SHAPE[1] // PATCH_SIZE)
