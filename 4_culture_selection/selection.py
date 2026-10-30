#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
selection.py   (STEP 4)
============

Keeps only the cultures whose two chambers became functionally coupled, and
copies their valid recordings to the analysis folder.

    Input : ../2_screening/recordings.csv     valid recordings (step 2)
    Output: cultures.csv                      coupling test of every culture
            ../5_analysis_data                recordings that are analysed

Coupling test
-------------
For each culture, we count the bursts that coincide between chambers (within
the coincidence window chosen in step 3), summed over all its valid recordings.
This number is compared with the one expected by chance, obtained by randomly
shifting the bursts of one chamber (circular time-shift surrogates, see
common/bursts.py). The culture is kept if p < ALPHA.

Run:
    python3 selection.py        (requires numpy and pandas)
"""

import os
import sys
import shutil

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True  # do not create __pycache__ folders

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import bursts
import recordings


# =============================================================================
# CONFIGURATION
# =============================================================================

# Coincidence window (s), chosen in step 3 (3_window/window.py).
COINCIDENCE_WINDOW = 1.0

# Number of surrogate repetitions and significance level.
N_REPETITIONS = 10000
ALPHA = 0.05

# Source chamber of each culture (the one connected to the wide end of the
# funnel-shaped microchannels). The other chamber is the target.
SOURCE_CHAMBER = {
    "20_02": "B",
    "20_04": "A",
    "30_04_1": "A",
    "30_04_2": "B",
    "30_04_3": "B",
    "15_05_1": "A",
    "15_05_2": "B",
    "22_05": "B",
}

# Fixed random seed so that the results are reproducible.
rng = np.random.default_rng(12345)


# =============================================================================
# 1. COUPLING TEST FOR EACH CULTURE
# =============================================================================

def test_culture(paths):
    """
    Observed and chance coincidences for one culture (all its recordings summed).

    Returns (observed, chance_mean, p_value).
    """
    observed = 0
    chance = np.zeros(N_REPETITIONS)

    for path in paths:
        data = bursts.detect_chamber_bursts(path)
        A, B = data["bursts_A"], data["bursts_B"]

        A_to_B, B_to_A, _, _ = bursts.match_bursts(A, B, COINCIDENCE_WINDOW)
        observed += A_to_B + B_to_A
        chance += bursts.chance_matches(A, B, data["duration"], COINCIDENCE_WINDOW, N_REPETITIONS, rng)

    # p-value: fraction of surrogates with at least as many coincidences as observed
    p_value = (np.sum(chance >= observed) + 1) / (N_REPETITIONS + 1)
    return observed, float(chance.mean()), p_value


def test_all_cultures(paths):
    """Run the coupling test on every culture. Returns a DataFrame."""
    by_culture = {}
    for path in paths:
        culture = recordings.culture_name(recordings.recording_name(path))
        by_culture.setdefault(culture, []).append(path)

    rows = []
    for culture, culture_paths in sorted(by_culture.items()):
        observed, chance, p_value = test_culture(culture_paths)
        rows.append({
            "culture": culture,
            "n_valid_recordings": len(culture_paths),
            "coincident_bursts": observed,
            "expected_by_chance": round(chance, 2),
            "p_value": round(p_value, 4),
            "coupled": p_value < ALPHA,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 2. COPY THE RECORDINGS OF THE COUPLED CULTURES
# =============================================================================

def copy_recordings(paths, coupled_cultures):
    """Copy the recordings of the coupled cultures and write index.csv."""
    # (If the selection changes, delete 5_analysis_data by hand first.)
    os.makedirs(recordings.ANALYSIS_DATA, exist_ok=True)

    index_rows = []
    for path in paths:
        name = recordings.recording_name(path)
        culture = recordings.culture_name(name)
        if culture not in coupled_cultures:
            continue

        # --- Copy the recording folder (spikes and ROI files) ---
        folder = os.path.dirname(os.path.relpath(path, recordings.RAW_DATA))
        shutil.copytree(os.path.join(recordings.RAW_DATA, folder),
                        os.path.join(recordings.ANALYSIS_DATA, folder), dirs_exist_ok=True)

        # --- One row of the index ---
        source = SOURCE_CHAMBER[culture]
        index_rows.append({
            "culture": culture,
            "div": recordings.div_number(name),
            "recording": name,
            "source_chamber": source,
            "target_chamber": "B" if source == "A" else "A",
        })

    index = pd.DataFrame(index_rows).sort_values(["culture", "div"])
    index.to_csv(os.path.join(recordings.ANALYSIS_DATA, "index.csv"), index=False)
    return index


# =============================================================================
# MAIN
# =============================================================================

def main():
    paths = recordings.valid_recordings()

    cultures = test_all_cultures(paths)
    cultures.insert(1, "label", cultures["culture"].map(recordings.CULTURE_LABELS))
    cultures.to_csv(os.path.join(HERE, "cultures.csv"), index=False)

    coupled = set(cultures.loc[cultures["coupled"], "culture"])
    index = copy_recordings(paths, coupled)

    # --- Summary ---
    print(f"Coincidence window: {COINCIDENCE_WINDOW} s, {N_REPETITIONS} surrogates\n")
    print(cultures.to_string(index=False))
    print(f"\n{len(coupled)} coupled cultures, {len(index)} recordings copied to 5_analysis_data.")


if __name__ == "__main__":
    main()
