# -*- coding: utf-8 -*-
"""
recordings.py
=============

Small helpers to locate recordings and read their names.

Recording names have the form <culture>_DIV<day>, e.g. '15_05_1_DIV17':
    culture = '15_05_1'   (culture date day_month, numbered if several share a date)
    div     = 17          (days in vitro)
"""

import os
import re
import glob

import pandas as pd


# =============================================================================
# FOLDERS OF THE PROJECT
# =============================================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

RAW_DATA = os.path.join(ROOT, "1_raw_data")
SCREENING_TABLE = os.path.join(ROOT, "2_screening", "recordings.csv")
ANALYSIS_DATA = os.path.join(ROOT, "5_analysis_data")


# Labels of the cultures in the article (C1-C7: the analysed cultures, numbered by
# their number of inter-chamber propagations; C8: the culture excluded in step 4).
CULTURE_LABELS = {
    "15_05_1": "C1",
    "15_05_2": "C2",
    "30_04_2": "C3",
    "20_02": "C4",
    "30_04_3": "C5",
    "22_05": "C6",
    "30_04_1": "C7",
    "20_04": "C8",
}


# =============================================================================
# NAMES
# =============================================================================

def recording_name(path):
    """'.../15_05_1_DIV17_spikesData.csv' -> '15_05_1_DIV17'"""
    return os.path.basename(path).replace("_spikesData.csv", "")


def culture_name(recording):
    """'15_05_1_DIV17' -> '15_05_1'"""
    return re.sub(r"_DIV\d+$", "", recording)


def div_number(recording):
    """'15_05_1_DIV17' -> 17"""
    return int(re.search(r"DIV(\d+)", recording).group(1))


# =============================================================================
# FINDING RECORDINGS
# =============================================================================

def spike_files(folder):
    """All spikesData.csv files inside a data folder (<culture>/DIVxx/...)."""
    return sorted(glob.glob(os.path.join(folder, "*", "DIV*", "*_spikesData.csv")))


def valid_recordings():
    """
    Paths of the recordings that passed the screening (step 2).
    Requires 2_screening/screening.py to have been run.
    """
    table = pd.read_csv(SCREENING_TABLE)
    valid = table[table["valid"]]
    return [os.path.join(RAW_DATA, p) for p in valid["path"]]
