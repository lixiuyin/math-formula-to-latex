"""Headless-safe figure output: always save to file, show only when a GUI exists."""

import os

import matplotlib
import matplotlib.pyplot as plt

_NON_INTERACTIVE_BACKENDS = {'agg', 'pdf', 'ps', 'svg', 'cairo', 'template'}


def finalize_figure(filepath):
    """Save the current figure to a file, then show it if a GUI backend is
    available (on headless servers, e.g. over SSH, plt.show() is a no-op and
    the figure would otherwise be lost)."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {filepath}")
    if matplotlib.get_backend().lower() not in _NON_INTERACTIVE_BACKENDS:
        plt.show()
    plt.close()
