# -*- coding: utf-8 -*-
"""
plot_style.py
=============

Common style for every figure of the project.

To make the text of ALL figures bigger or smaller, change the sizes below and
run the scripts again. Nothing else needs to be changed.
"""

import matplotlib
matplotlib.use("Agg")           # save figures to files without opening windows
import matplotlib.pyplot as plt


# =============================================================================
# TEXT SIZES (points)  <-- edit here
# =============================================================================

AXIS_LABEL_SIZE = 20    # axis titles ("Number of bursts", "Culture", ...)
TICK_LABEL_SIZE = 18    # numbers / names on the axes
LEGEND_SIZE = 17        # legend text
ANNOTATION_SIZE = 18    # text written inside the plot (delta t, p-values, ...)


# =============================================================================
# FIGURE SIZE (inches)  <-- edit here
# =============================================================================
# All figures have the same height and the same margins, so that they can be
# placed side by side at the same height and the text has the same size in all.

FIG_HEIGHT = 5.5        # height of every figure
MARGIN_LEFT = 1.4       # space for the y-axis title and numbers
MARGIN_RIGHT = 0.3
MARGIN_BOTTOM = 1.25     # space for the x-axis title and (rotated) names
MARGIN_TOP = 0.3

# Figures of the main text (step 6) are stacked in one column: same width and
# height, short in y. They use these values instead of FIG_HEIGHT / MARGIN_LEFT.
COLUMN_WIDTH = 6.5
COLUMN_HEIGHT = 4.5
COLUMN_MARGIN_LEFT = 1.7    # two-line y-axis titles need a bit more space


# =============================================================================
# COLOURS
# =============================================================================

DATA_COLOUR = "0.25"        # observed data (dark grey)
EXCLUDED_COLOUR = "0.7"     # excluded data (light grey)
CHANCE_COLOUR = "#d62728"   # chance level (red)
WINDOW_COLOUR = "#1f77b4"   # chosen window (blue)
SOURCE_COLOUR = "#1f77b4"   # source chamber (blue)
TARGET_COLOUR = "#d62728"   # target chamber (red)
CULTURE_COLOUR = "0.75"     # individual cultures (light grey)


# =============================================================================
# FUNCTIONS
# =============================================================================

def apply():
    """Apply the common style. Call it before creating a figure."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": TICK_LABEL_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
    })


def new_figure(width, height=FIG_HEIGHT, left=MARGIN_LEFT):
    """
    Create a figure with the common margins (sizes in inches).
    By default all figures have the same height; only the width changes.
    """
    apply()
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=left / width, right=1 - MARGIN_RIGHT / width,
                        bottom=MARGIN_BOTTOM / height, top=1 - MARGIN_TOP / height)
    return fig, ax


def new_column_figure():
    """Figure of the main-text column: all of them exactly the same size."""
    return new_figure(COLUMN_WIDTH, COLUMN_HEIGHT, COLUMN_MARGIN_LEFT)


def save(fig, folder, name):
    """
    Save a figure as PNG (300 dpi) and PDF in `folder`.
    The figure is NOT cropped, so all figures keep exactly the same height.
    """
    import os
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(folder, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
