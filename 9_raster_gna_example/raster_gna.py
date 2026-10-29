#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
raster_gna.py   (STEP 9)
=============

Figure 2b of the article: raster plot of one recording and its global network
activity (GNA), with the burst-detection threshold and one inter-burst interval (IBI).

    Input : ../1_raw_data/15_05_1/DIV21/15_05_1_DIV21_spikesData.csv          spikes (NETCAL)
            ../software_outputs/VisualizeFront/15_05_1/DIV21/
                15_05_1_DIV21_Global_activity.csv                             GNA computed by VisualizeFront
    Output: raster_gna.png / .pdf

The GNA was computed with VisualizeFront (analysis software of the laboratory, not
included here): fraction of the active ROIs that fire within a 1 s sliding window
(window size and threshold in ../software_outputs/VisualizeFront/.../*_parameters_used.txt).
Network bursts are the episodes in which the GNA exceeds 0.2.

As in VisualizeFront, repeated spikes of the same ROI within 0.5 s are removed
before drawing the raster (only the first spike is kept).

Run:
    python3 raster_gna.py      (requires numpy, pandas and matplotlib)
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


# =============================================================================
# CONFIGURATION
# =============================================================================

RECORDING = "15_05_1_DIV21"            # culture C1, DIV 21
SPIKES = os.path.join(ROOT, "1_raw_data", "15_05_1", "DIV21", f"{RECORDING}_spikesData.csv")
GNA = os.path.join(ROOT, "software_outputs", "VisualizeFront", "15_05_1", "DIV21",
                   f"{RECORDING}_Global_activity.csv")

THRESHOLD = 0.2        # burst threshold on the GNA (dashed line)
MERGE_GAP = 0.4        # above-threshold segments closer than this (s) are one burst
REPETITION_WINDOW = 0.5  # repeated spikes of one ROI within this window (s) are removed
IBI_PAIR = (0, 1)      # the IBI arrow is drawn between these two bursts

# Text sizes and figure shape (width / height of the final image)
LABEL_SIZE, TICK_SIZE, IBI_SIZE = 30, 24, 26
AXIS_WIDTH, TICK_LENGTH = 1.8, 7
ASPECT = 1.5


# =============================================================================
# 1. DATA
# =============================================================================

def load_spikes(path):
    """ROI index and time (s) of every spike."""
    data = pd.read_csv(path, skipinitialspace=True)
    data.columns = [c.strip().lower() for c in data.columns]
    return data[data.columns[0]].to_numpy(int), data[data.columns[1]].to_numpy(float)


def remove_repetitions(ids, times, window):
    """Keep only the first spike of each ROI within `window` s (as in VisualizeFront)."""
    keep = np.zeros(len(times), bool)
    last_id, last_time = None, -np.inf
    for k in np.lexsort((times, ids)):          # sorted by ROI, then by time
        if ids[k] != last_id or times[k] - last_time > window:
            keep[k] = True
            last_id, last_time = ids[k], times[k]
    return ids[keep], times[keep]


def load_gna(path):
    """Time (s) and GNA value, as exported by VisualizeFront."""
    gna = pd.read_csv(path, header=None).to_numpy()
    return gna[:, 0], gna[:, 1]


def detect_bursts(t, gna, threshold, merge_gap):
    """Bursts = segments with GNA >= threshold (merged if closer than merge_gap).
    Returns a list of [start, end, peak time]."""
    above = np.where(gna >= threshold)[0]
    segments = np.split(above, np.where(np.diff(above) > 1)[0] + 1)
    bursts = [[t[s[0]], t[s[-1]], t[s[np.argmax(gna[s])]]] for s in segments if len(s)]
    merged = [bursts[0]]
    for b in bursts[1:]:
        if b[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    return merged


# =============================================================================
# 2. FIGURE
# =============================================================================

def plot(ids, times, t, gna, bursts):
    plt.rcParams.update({"font.family": "sans-serif"})
    fig, (ax_r, ax_g) = plt.subplots(2, 1, figsize=(8.6, 7.5), sharex=True,
                                     gridspec_kw=dict(height_ratios=[3, 1.2], hspace=0.20))

    # --- Raster (ROI 1 at the top, as in the image of the culture) ---
    ax_r.scatter(times, ids, s=1.4, c="#1f2a44", marker="|", linewidths=0.6)
    ax_r.set_ylabel("ROI", fontsize=LABEL_SIZE)
    ax_r.set_ylim(405, 0)
    ax_r.set_yticks([0, 100, 200, 300, 400])

    # --- GNA and threshold ---
    ax_g.fill_between(t, 0, gna, color="#8fb030", lw=0)
    ax_g.plot(t, gna, color="#5f7a18", lw=1.0)
    ax_g.axhline(THRESHOLD, color="#b03030", lw=1.6, ls="--")
    ax_g.set_ylabel("GNA", fontsize=LABEL_SIZE)
    ax_g.set_xlabel("Time (s)", fontsize=LABEL_SIZE)
    ax_g.set_ylim(0, 1.02)
    ax_g.set_yticks([0, 0.5, 1.0])
    ax_g.set_xticks(range(0, 601, 100))

    # --- IBI between two consecutive bursts (peak to peak) ---
    i, j = IBI_PAIR
    p1, p2, y = bursts[i][2], bursts[j][2], 0.9
    ax_g.annotate("", xy=(p2, y), xytext=(p1, y), arrowprops=dict(arrowstyle="<->", color="k", lw=2))
    ax_g.text((p1 + p2) / 2, y - 0.1, "IBI", ha="center", va="top", fontsize=IBI_SIZE)

    fig.align_ylabels((ax_r, ax_g))
    for ax in (ax_r, ax_g):
        ax.tick_params(labelsize=TICK_SIZE, width=AXIS_WIDTH, length=TICK_LENGTH)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(AXIS_WIDTH)

    # --- Final image with width / height = ASPECT ---
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = bbox.width, bbox.height
    fig.set_size_inches(fig.get_size_inches()[0], fig.get_size_inches()[1] + (w / ASPECT - h))

    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"raster_gna.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ids, times = load_spikes(SPIKES)
    n_before = len(times)
    ids, times = remove_repetitions(ids, times, REPETITION_WINDOW)
    t, gna = load_gna(GNA)
    bursts = detect_bursts(t, gna, THRESHOLD, MERGE_GAP)
    plot(ids, times, t, gna, bursts)

    i, j = IBI_PAIR
    print(f"{RECORDING} (C1): {len(times)} of {n_before} spikes kept after removing repetitions")
    print(f"{len(bursts)} network bursts (GNA >= {THRESHOLD}); "
          f"IBI shown = {bursts[j][2] - bursts[i][2]:.1f} s")


if __name__ == "__main__":
    main()
