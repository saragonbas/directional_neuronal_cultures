#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
window.py   (STEP 3)
=========

Chooses the coincidence window (delta t): the maximum time between a burst in
one chamber and a burst in the other for the pair to be considered the same
event propagating between chambers.

The window is chosen from all valid recordings (step 2), before any culture is
selected, so that the culture selection (step 4) can use it without circularity.

    Input : ../2_screening/recordings.csv       list of valid recordings
    Output: window_separations.png / .pdf       figure: observed vs chance separations
            window_captured.png / .pdf          figure: propagations captured vs window
            window_choice.csv                   numbers behind the figures

Procedure
---------
1. For every burst of chamber A, measure the time to the nearest burst of
   chamber B (all recordings pooled).
2. Repeat the same measurement after randomly shifting the bursts of chamber B
   (circular time-shift, see common/bursts.py). This gives the distribution
   expected by chance.
3. Real propagations appear as an excess of short separations over chance.
   The chosen window is the one that captures (almost) all of this excess.

Run:
    python3 window.py           (requires numpy, pandas and matplotlib)
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True  # do not create __pycache__ folders

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import bursts
import recordings
import plot_style


# =============================================================================
# CONFIGURATION
# =============================================================================

# Separations are studied up to this value (s), in bins of this width (s).
MAX_SEPARATION = 3.0
BIN_WIDTH = 0.1

# Number of surrogate repetitions per recording.
N_REPETITIONS = 200

# Window used in the analysis (marked in the figure).
CHOSEN_WINDOW = 1.0

# Fixed random seed so that the results are reproducible.
rng = np.random.default_rng(2026)


# =============================================================================
# 1. SEPARATION BETWEEN EACH BURST AND THE NEAREST BURST OF THE OTHER CHAMBER
# =============================================================================

def nearest_separations(bursts_A, bursts_B):
    """For each burst of A, time (s) to the nearest burst of B."""
    bursts_B = np.asarray(bursts_B)
    return [float(np.min(np.abs(bursts_B - t))) for t in bursts_A]


def collect_separations():
    """
    Returns
    -------
    observed  : separations in the real data (all recordings pooled)
    surrogate : list with one array of separations per surrogate repetition
                (all recordings pooled within each repetition)
    n_recordings
    """
    paths = recordings.valid_recordings()

    observed = []
    surrogate = [[] for _ in range(N_REPETITIONS)]

    for path in paths:
        data = bursts.detect_chamber_bursts(path)
        A, B, duration = data["bursts_A"], data["bursts_B"], data["duration"]

        # --- Real data ---
        observed += nearest_separations(A, B)

        # --- Surrogates: shift all bursts of B by the same random offset ---
        for k in range(N_REPETITIONS):
            offset = rng.uniform(5, duration - 5)
            shifted_B = (np.asarray(B) + offset) % duration
            surrogate[k] += nearest_separations(A, shifted_B)

    return np.array(observed), [np.array(s) for s in surrogate], len(paths)


# =============================================================================
# 2. HISTOGRAMS AND EXCESS OVER CHANCE
# =============================================================================

def compare_with_chance(observed, surrogate):
    """
    Histogram of the observed separations and of the separations expected by
    chance (mean and 95% interval over surrogate repetitions).

    The excess over chance, accumulated from 0 to each separation, tells how
    many real propagations a window of that width captures. It is expressed as
    a percentage of its maximum.
    """
    edges = np.arange(0, MAX_SEPARATION + BIN_WIDTH / 2, BIN_WIDTH)
    centres = 0.5 * (edges[:-1] + edges[1:])

    observed_counts = np.histogram(observed, edges)[0]
    surrogate_counts = np.array([np.histogram(s, edges)[0] for s in surrogate])

    chance_mean = surrogate_counts.mean(axis=0)
    chance_low = np.percentile(surrogate_counts, 2.5, axis=0)
    chance_high = np.percentile(surrogate_counts, 97.5, axis=0)

    cumulative_excess = np.cumsum(observed_counts - chance_mean)
    captured_pct = 100 * cumulative_excess / cumulative_excess.max()

    return pd.DataFrame({
        "separation_bin_start_s": edges[:-1],
        "separation_bin_end_s": edges[1:],
        "separation_bin_centre_s": centres,
        "observed": observed_counts,
        "chance_mean": chance_mean,
        "chance_2.5pct": chance_low,
        "chance_97.5pct": chance_high,
        "propagations_captured_pct": captured_pct,
    })


# =============================================================================
# 3. FIGURES
# =============================================================================

def plot_separations(table):
    """
    Fig. SF3a: distribution of the time between each burst of one chamber and the
    nearest burst of the other chamber, compared with chance.
    """
    fig, ax = plot_style.new_figure(width=8)
    x = table["separation_bin_centre_s"]

    # --- Chance level (mean and 95% interval) ---
    ax.fill_between(x, table["chance_2.5pct"], table["chance_97.5pct"], step="mid",
                    color=plot_style.CHANCE_COLOUR, alpha=0.25, lw=0, zorder=1)
    ax.step(x, table["chance_mean"], where="mid", color=plot_style.CHANCE_COLOUR, lw=2.5,
            label="Expected by chance", zorder=3)

    # --- Observed data ---
    ax.bar(x, table["observed"], width=BIN_WIDTH * 0.85, color=plot_style.DATA_COLOUR,
           label="Observed", zorder=2)

    # --- Chosen window ---
    ax.axvline(CHOSEN_WINDOW, color=plot_style.WINDOW_COLOUR, ls="--", lw=2.2, zorder=4)
    ax.text(CHOSEN_WINDOW + 0.06, table["observed"].max() * 0.5,
            f"\u0394t = {CHOSEN_WINDOW:g} s", color=plot_style.WINDOW_COLOUR,
            fontsize=plot_style.ANNOTATION_SIZE)

    ax.set_xlim(0, MAX_SEPARATION)
    ax.set_xlabel("Time to nearest burst\nin the other chamber (s)")
    ax.set_ylabel("Number of bursts")
    ax.legend(frameon=False, loc="upper right")
    plot_style.save(fig, HERE, "window_separations")


def plot_captured(table):
    """
    Fig. SF3b: percentage of the inter-chamber propagations (excess over chance)
    captured as a function of the coincidence window.
    """
    fig, ax = plot_style.new_figure(width=8)
    ends = table["separation_bin_end_s"]
    captured = table["propagations_captured_pct"]

    ax.plot(ends, captured, "-o", color=plot_style.DATA_COLOUR, ms=6, lw=2.5)
    ax.axhline(100, color="0.6", lw=1.2, ls=":")

    # --- Chosen window ---
    i = int(round(CHOSEN_WINDOW / BIN_WIDTH)) - 1
    ax.axvline(CHOSEN_WINDOW, color=plot_style.WINDOW_COLOUR, ls="--", lw=2.2)
    ax.plot(CHOSEN_WINDOW, captured.iloc[i], "o", color=plot_style.WINDOW_COLOUR, ms=12, zorder=5)
    ax.text(CHOSEN_WINDOW + 0.12, captured.iloc[i] - 25,
            f"\u0394t = {CHOSEN_WINDOW:g} s\n{captured.iloc[i]:.0f}% captured",
            color=plot_style.WINDOW_COLOUR, fontsize=plot_style.ANNOTATION_SIZE)

    ax.set_xlim(0, MAX_SEPARATION)
    ax.set_ylim(0, 108)
    ax.set_xlabel("Coincidence window \u0394t (s)")
    ax.set_ylabel("Propagations captured (%)")
    plot_style.save(fig, HERE, "window_captured")


# =============================================================================
# MAIN
# =============================================================================

def main():
    observed, surrogate, n_recordings = collect_separations()
    table = compare_with_chance(observed, surrogate)
    table.round(3).to_csv(os.path.join(HERE, "window_choice.csv"), index=False)
    plot_separations(table)
    plot_captured(table)

    # --- Summary ---
    print(f"{n_recordings} recordings, {len(observed)} bursts in chamber A.\n")
    print("window (s)   propagations captured   observed pairs   pairs expected by chance")
    for window in (0.5, 1.0, 1.5, 2.0):
        i = int(round(window / BIN_WIDTH)) - 1
        print(f"{window:8.1f}   {table['propagations_captured_pct'].iloc[i]:18.0f}%   "
              f"{table['observed'].iloc[:i + 1].sum():14d}   {table['chance_mean'].iloc[:i + 1].sum():20.1f}")


if __name__ == "__main__":
    main()
