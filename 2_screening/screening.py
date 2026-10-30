#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
screening.py   (STEP 2)
=============

Checks every recording and decides whether it is valid.

    Input : ../1_raw_data        all baseline recordings
    Output: recordings.csv       every recording: bursts detected in each chamber,
                                 valid or not, and the reason if not

A recording is discarded if:
    a) it is in the list of manual exclusions (reason given below),
    b) it was recorded after DIV_MAX,
    c) one of the two chambers has no bursts.

Run:
    python3 screening.py        (requires numpy and pandas)
"""

import os
import sys

import pandas as pd

sys.dont_write_bytecode = True  # do not create __pycache__ folders

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import bursts
import recordings


# =============================================================================
# CONFIGURATION
# =============================================================================

# Recordings excluded after inspection, with the reason.
MANUAL_EXCLUSIONS = {
    "20_04_DIV7": "wrong frame rate in NETCAL (3.33 instead of 33.33 fps): spike times are incorrect",
    "30_04_1_DIV11": "abnormally high activity (outlier, ~4 times the usual firing)",
}

# Only one culture was recorded after DIV 24, so later days are not used.
DIV_MAX = 24


# =============================================================================
# SCREENING
# =============================================================================

def exclusion_reason(recording, burst_data):
    """Return why a recording is excluded, or '' if it is valid."""
    if recording in MANUAL_EXCLUSIONS:
        return MANUAL_EXCLUSIONS[recording]
    if recordings.div_number(recording) > DIV_MAX:
        return f"recorded after DIV {DIV_MAX}"
    if not burst_data["bursts_A"] or not burst_data["bursts_B"]:
        return "no bursts in one of the chambers"
    return ""


def main():
    rows = []
    for path in recordings.spike_files(recordings.RAW_DATA):
        name = recordings.recording_name(path)
        result = bursts.detect_chamber_bursts(path)
        reason = exclusion_reason(name, result)

        rows.append({
            "culture": recordings.culture_name(name),
            "div": recordings.div_number(name),
            "recording": name,
            "active_neurons_A": result["active_A"],
            "active_neurons_B": result["active_B"],
            "bursts_A": len(result["bursts_A"]),
            "bursts_B": len(result["bursts_B"]),
            "duration_s": round(result["duration"], 1),
            "valid": reason == "",
            "exclusion_reason": reason,
            "path": os.path.relpath(path, recordings.RAW_DATA),
        })

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(HERE, "recordings.csv"), index=False)

    # --- Summary ---
    print(f"Recordings: {len(table)} in total, {table['valid'].sum()} valid "
          f"({table.loc[table['valid'], 'culture'].nunique()} cultures).\n")
    print("Excluded recordings:")
    print(table.loc[~table["valid"], ["recording", "exclusion_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
