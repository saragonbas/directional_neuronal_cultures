#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
threshold_check.py
==================

Compares the chamber-specific burst threshold used in the analysis (0.1) with
the threshold used for the whole network (0.2).

For each threshold, over all valid recordings (step 2):
  - number of bursts detected in each chamber,
  - number of coincident bursts between chambers (window = 1 s),
  - number of coincidences expected by chance (circular time-shift surrogates),
  - excess over chance = real propagations.

    Input : ../2_screening/recordings.csv   list of valid recordings
    Output: threshold_check.csv

Run:
    python3 threshold_check.py      (requires numpy and pandas)
"""

import os
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import bursts
import recordings


# =============================================================================
# CONFIGURATION
# =============================================================================

THRESHOLDS = [0.1, 0.2]
WINDOW = 1.0
N_REPETITIONS = 1000
rng = np.random.default_rng(7)


# =============================================================================
# COUNT BURSTS AND PROPAGATIONS FOR ONE THRESHOLD
# =============================================================================

def count_for_threshold(threshold, paths):
    """Totals over all recordings for a given chamber burst threshold."""
    bursts.BURST_THRESHOLD = threshold   # change the threshold used by bursts.py

    totals = {"threshold": threshold, "bursts_A": 0, "bursts_B": 0,
              "coincident": 0, "expected_by_chance": 0.0}

    for path in paths:
        data = bursts.detect_chamber_bursts(path)
        A, B = data["bursts_A"], data["bursts_B"]

        A_to_B, B_to_A, _, _ = bursts.match_bursts(A, B, WINDOW)
        chance = bursts.chance_matches(A, B, data["duration"], WINDOW, N_REPETITIONS, rng)

        totals["bursts_A"] += len(A)
        totals["bursts_B"] += len(B)
        totals["coincident"] += A_to_B + B_to_A
        totals["expected_by_chance"] += float(chance.mean())

    totals["real_propagations"] = totals["coincident"] - totals["expected_by_chance"]
    return totals


# =============================================================================
# MAIN
# =============================================================================

def main():
    paths = recordings.valid_recordings()
    table = pd.DataFrame([count_for_threshold(t, paths) for t in THRESHOLDS]).round(1)
    table.to_csv(os.path.join(HERE, "threshold_check.csv"), index=False)

    print(f"{len(paths)} recordings, coincidence window = {WINDOW} s\n")
    print(table.to_string(index=False))

    low, high = table.iloc[0], table.iloc[1]
    lost = 100 * (1 - high["real_propagations"] / low["real_propagations"])
    print(f"\nWith threshold {high['threshold']}, {lost:.0f}% of the real propagations "
          f"detected with {low['threshold']} are lost.")


if __name__ == "__main__":
    main()
