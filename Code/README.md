# ViT-based Image-to-LaTeX

Handwritten/rendered mathematical expression recognition with a Vision
Transformer encoder and a Transformer decoder, generating LaTeX token
sequences (PyTorch). Supports two datasets: printed `im2latex-100k`
(default) and handwritten `MathWriting-human`.

## Project Structure

```
Code/
├── main.py                      # Entry point: train + evaluate pipeline
├── requirements.txt
├── outputs/                     # Run artifacts (created automatically, git-ignored)
│   ├── checkpoints/             # transformer_model_best.pt, transformer_model.pt
│   ├── figures/                 # training_history.png, predictions.png
│   └── demo/                    # demo_*.png from the interactive demo
├── tests/                       # pytest suite (tokenizer, metrics, scheduler, model)
└── vit_latex/
    ├── config.py                # All hyperparameters + DATASETS registry
    ├── plotting.py              # Headless-safe figure saving
    ├── data/
    │   ├── tokenizer.py         # Vocab (pad=0, [UNK]=1) + raw-LaTeX regex tokenizer
    │   └── preprocessing.py     # Image preprocessing + HF dataset loading
    ├── models/
    │   ├── encoder.py           # ViT encoder (patches, position emb., pre-LN blocks)
    │   ├── decoder.py           # Decoder layers + TokenOutput (prior bias)
    │   └── captioner.py         # Full model + greedy / beam search decoding
    ├── training/
    │   ├── metrics.py           # Masked loss / accuracy (padding ignored)
    │   ├── scheduler.py         # AdamW + Transformer warmup LR schedule
    │   └── trainer.py           # Training loop, early stopping, checkpointing
    └── evaluation/
        └── evaluator.py         # Evaluation, BLEU, visualization, demo
```

## Setup

Python 3.11+ required (see `pyproject.toml` / `.python-version`).

```bash
pip install -r requirements.txt   # or: uv pip install -r requirements.txt
```

## Usage

```bash
python main.py                                   # train + evaluate on im2latex
python main.py --dataset mathwriting             # handwritten dataset
python main.py --beam                            # add greedy-vs-beam BLEU comparison
```

The first run downloads the dataset from Hugging Face. Training uses GPU
automatically when available. Artifacts for non-default datasets get a
`_<dataset>` filename suffix (e.g. `transformer_model_mathwriting.pt`)
so runs do not overwrite each other.

To evaluate a trained model without retraining:

```bash
python main.py --eval-only [--dataset mathwriting] [--beam]
python main.py --eval-only --weights outputs/checkpoints/transformer_model.pt
```

## Tests

```bash
pip install pytest   # or: uv sync (installs the dev group)
pytest
```

## References

- Dosovitskiy et al., 2021. An Image is Worth 16x16 Words (ViT).
- Vaswani et al., 2017. Attention Is All You Need (warmup schedule Sec. 5.3,
  beam search settings Sec. 6.1).
