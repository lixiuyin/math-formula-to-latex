"""Render the model architecture diagram used in the README (assets/architecture.png).

Self-contained matplotlib script (no project imports). Run from the repo root:
    Code/.venv/bin/python assets/architecture_diagram.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_PATH = "assets/architecture.png"

# Palette: data boxes (gray), encoder (blue), decoder (green), output (orange)
GRAY = ("#eeeeee", "#555555")
BLUE = ("#cfe8fc", "#1f77b4")
GREEN = ("#d7f0d3", "#2ca02c")
ORANGE = ("#ffe4c4", "#ff7f0e")

BOX_LEFT, BOX_RIGHT = 1.0, 9.0
CENTER = (BOX_LEFT + BOX_RIGHT) / 2

TOP_PAD = 0.45        # box top -> title baseline
LINE_GAP = 0.5        # spacing between text lines
BOT_PAD = 0.35        # last line -> box bottom
ARROW_GAP = 0.95      # vertical space between boxes


def box(ax, top_y, colors, lines, title_size=12, body_size=9.5):
    """Draw a rounded box whose top edge sits at top_y; return its bottom y."""
    face, edge = colors
    n = len(lines)
    height = TOP_PAD + LINE_GAP * (n - 1) + BOT_PAD if n > 1 else 0.85
    bottom = top_y - height
    ax.add_patch(
        FancyBboxPatch(
            (BOX_LEFT, bottom), BOX_RIGHT - BOX_LEFT, height,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.8, edgecolor=edge, facecolor=face, zorder=2,
        )
    )
    if n == 1:
        ax.text(CENTER, top_y - height / 2, lines[0], ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=edge, zorder=3)
    else:
        ax.text(CENTER, top_y - TOP_PAD, lines[0], ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=edge, zorder=3)
        for i, line in enumerate(lines[1:], start=1):
            ax.text(CENTER, top_y - TOP_PAD - i * LINE_GAP, line, ha="center",
                    va="center", fontsize=body_size, color="#222222", zorder=3)
    return bottom


def arrow(ax, y_from, y_to, label=None):
    ax.annotate("", xy=(CENTER, y_to), xytext=(CENTER, y_from),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.8), zorder=1)
    if label:
        ax.text(CENTER, (y_from + y_to) / 2, label, ha="center", va="center",
                fontsize=8, style="italic", color="#444444", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


def main():
    fig, ax = plt.subplots(figsize=(7.6, 9.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.3, 13.0)
    ax.axis("off")

    ax.text(CENTER, 12.7, "Image-to-LaTeX:  ViT encoder + Transformer decoder",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#111111")

    y = box(ax, 12.1, GRAY, ["Input image  ·  1 x 50 x 200  (grayscale, [0,1])"],
            title_size=11)
    arrow(ax, y, y - ARROW_GAP, "10x10 patches -> 100 tokens, linear proj (256-d) + pos emb")

    y = box(ax, y - ARROW_GAP, BLUE,
            ["ViT Encoder  x 8", "pre-LN · 4 heads (dim 64) · GELU MLP [512, 256]"])
    arrow(ax, y, y - ARROW_GAP, "image features  (B, 100, 256)")

    y = box(ax, y - ARROW_GAP, GREEN,
            ["Transformer Decoder  x 4",
             "post-LN · 8 heads (dim 32)",
             "(1)  causal self-attention  (over generated LaTeX tokens)",
             "(2)  cross-attention  (Q = text,  K / V = image features)",
             "(3)  feed-forward  (ReLU, 256 -> 512 -> 256)"])
    arrow(ax, y, y - ARROW_GAP)

    y = box(ax, y - ARROW_GAP, ORANGE,
            ["TokenOutput",
             "Linear(256 -> vocab) + log-frequency prior bias",
             "('' and [UNK] tokens banned)"],
            body_size=9)
    arrow(ax, y, y - ARROW_GAP, "decoding:  greedy / sampling / beam search (k=4, GNMT penalty)")

    y = box(ax, y - ARROW_GAP, GRAY, ["LaTeX token sequence"], title_size=11)

    ax.text(CENTER, y - 0.55, "7.73M parameters  =  ViT encoder 4.27M  +  decoder 3.46M",
            ha="center", va="center", fontsize=9, color="#666666")

    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
