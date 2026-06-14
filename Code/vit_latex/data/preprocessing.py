"""Image preprocessing and dataset loading via Hugging Face datasets.

Supported datasets (see config.DATASETS):
  - im2latex (yuntian-deng/im2latex-100k): printed formulas,
    space-separated 'formula' field
  - mathwriting (deepcopy/MathWriting-human): handwritten formulas,
    raw 'latex' field tokenized via tokenize_latex
"""

import numpy as np
import torch
from PIL import Image
from datasets import load_dataset as hf_load_dataset

from vit_latex.config import IMG_SHAPE, DATASET_SIZE, DATASETS, DEFAULT_DATASET
from vit_latex.data.tokenizer import tokenize_latex


def _dataset_config(dataset):
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'; available: {sorted(DATASETS)}")
    return DATASETS[dataset]


def _format_formula(formula, pretokenized):
    if not pretokenized:
        formula = tokenize_latex(formula)
    return "[START] " + formula + " [END]"


def preprocess_image(image):
    """Preprocess a single image: grayscale, resize to model input, scale to [0, 1].

    Accepts PIL.Image / numpy array / tensor and returns a float32 tensor of
    shape (1, H, W) (PyTorch channels-first).
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if not isinstance(image, Image.Image):
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)

    image = image.convert('L').resize(
        (IMG_SHAPE[1], IMG_SHAPE[0]), Image.BILINEAR)  # PIL.resize takes (width, height)
    arr = np.asarray(image, dtype=np.float32) / 255.0  # scale to [0, 1]
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


def load_formulas(n_samples=DATASET_SIZE, split="train", dataset=DEFAULT_DATASET):
    """Load only the LaTeX formulas (no image decoding/preprocessing).

    Much faster than load_dataset; sufficient for fitting the tokenizer,
    e.g. in evaluation-only runs.
    """
    cfg = _dataset_config(dataset)
    ds = hf_load_dataset(cfg['name'], split=cfg['splits'][split])
    if n_samples is not None and n_samples < len(ds):
        ds = ds.select(range(n_samples))
    return [_format_formula(f, cfg['pretokenized'])
            for f in ds[cfg['formula_field']]]


def load_dataset(n_samples=DATASET_SIZE, split="train", dataset=DEFAULT_DATASET):
    """Load a dataset from Hugging Face and preprocess images.

    Returns:
        images: float32 tensor of shape (N, 1, H, W)
        latex_texts: list of strings of the form "[START] ... [END]"
    """
    cfg = _dataset_config(dataset)
    ds = hf_load_dataset(cfg['name'], split=cfg['splits'][split])
    if n_samples is not None and n_samples < len(ds):
        ds = ds.select(range(n_samples))

    images = []
    latex_texts = []
    for example in ds:
        images.append(preprocess_image(example["image"]))
        latex_texts.append(
            _format_formula(example[cfg['formula_field']], cfg['pretokenized']))

    return torch.stack(images), latex_texts


def prepare_training_data(images, latex_texts, tokenizer):
    """Convert LaTeX texts to fixed-length token sequences (LongTensor)."""
    sequences = tokenizer(latex_texts)
    return images, sequences
