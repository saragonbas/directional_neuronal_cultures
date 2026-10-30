#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
directionality.py   (STEP 6)
=================

Direction of the propagation between chambers in the selected cultures.

    Input : ../5_analysis_data                     recordings of the coupled cultures
    Output: per_recording.csv                      counts for every recording
            per_culture.csv                        counts, DI, recruitment and binomial test per culture
            per_culture_mature.csv                 same, only recordings from DIV 14 on (recruitment score)
            per_stage.csv                          counts and DI per culture and maturation stage
            propagations_per_culture.png / .pdf    Fig. 5f: S->T vs T->S in every culture
            recruitment_score.png / .pdf           Fig. 5e: recruitment score (DIV 14-24), source vs target
            directionality_vs_div.png / .pdf       Fig. 5g: directionality index across maturation stages

Measures (for every recording, Delta t = 1 s)
--------
    n(S->T), n(T->S)   bursts that propagated source->target and target->source
    n(local,S/T)       bursts of each chamber that did not propagate

    Directionality index   DI   = [n(S->T) - n(T->S)] / [n(S->T) + n(T->S)]
    Recruitment score      R(S) = n(S->T) / [n(S->T) + n(local,S)]
                           R(T) = n(T->S) / [n(T->S) + n(local,T)]

Statistical unit
----------------
The culture is the statistical unit: counts are summed over all recordings of a
culture, so that a culture with many recordings does not dominate.
Within each culture, S->T vs T->S is compared with an exact two-sided binomial test.
The recruitment score (4 mature cultures) and the time course are descriptive.

Run:
    python3 directionality.py      (requires numpy, pandas and matplotlib)
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

# Coincidence window (s), chosen in step 3 (3_window/window.py).
COINCIDENCE_WINDOW = 1.0

# Maturation stages (DIV ranges) used in the time-course figure.
# Note: the last stage (DIV 18-24) contains only 2 cultures.
STAGES = [(5, 8), (10, 12), (14, 17), (18, 24)]

# Culture labels (C1-C8) are defined in common/recordings.py.
CULTURE_LABELS = recordings.CULTURE_LABELS

# The recruitment score is computed in mature cultures, i.e. the last two stages of
# the time course (DIV 14-24), around the example recording (C1, DIV 17).
# Only 4 cultures were recorded at these ages, so this result is descriptive.
MATURE_FROM_DIV = 14

# A culture is shown in a stage only if it has at least this many propagations
# there (with fewer, the DI is not meaningful: e.g. 1 propagation gives DI = +-1).
MIN_PROPAGATIONS = 3


# =============================================================================
# 1. COUNT PROPAGATIONS IN EVERY RECORDING
# =============================================================================

def count_recording(path, source_chamber):
    """
    Number of propagations in each direction and of local bursts in each chamber,
    with chamber A/B translated into source (S) / target (T).
    """
    data = bursts.detect_chamber_bursts(path)
    A_to_B, B_to_A, local_A, local_B = bursts.match_bursts(
        data["bursts_A"], data["bursts_B"], COINCIDENCE_WINDOW)

    if source_chamber == "A":
        return {"S_to_T": A_to_B, "T_to_S": B_to_A, "local_S": local_A, "local_T": local_B}
    return {"S_to_T": B_to_A, "T_to_S": A_to_B, "local_S": local_B, "local_T": local_A}


def count_all_recordings():
    """Counts for every recording listed in 5_analysis_data/index.csv."""
    index = pd.read_csv(os.path.join(recordings.ANALYSIS_DATA, "index.csv"))
    rows = []
    for _, rec in index.iterrows():
        path = os.path.join(recordings.ANALYSIS_DATA, rec["culture"], f"DIV{rec['div']:02d}",
                            f"{rec['recording']}_spikesData.csv")
        counts = count_recording(path, rec["source_chamber"])
        rows.append({"culture": rec["culture"], "div": rec["div"],
                     "recording": rec["recording"], **counts})
    return pd.DataFrame(rows)


# =============================================================================
# 2. MEASURES
# =============================================================================

def add_measures(table):
    """Add DI, R(S) and R(T) to a table of summed counts."""
    table = table.copy()
    propagations = table["S_to_T"] + table["T_to_S"]
    table["n_propagations"] = propagations
    table["DI"] = (table["S_to_T"] - table["T_to_S"]) / propagations.replace(0, np.nan)
    table["R_source"] = table["S_to_T"] / (table["S_to_T"] + table["local_S"]).replace(0, np.nan)
    table["R_target"] = table["T_to_S"] / (table["T_to_S"] + table["local_T"]).replace(0, np.nan)
    return table


def stage_of(div):
    """Name of the maturation stage of a DIV, e.g. 'DIV 10-12'."""
    for low, high in STAGES:
        if low <= div <= high:
            return f"DIV {low}–{high}"
    return None


# =============================================================================
# 3. STATISTICS
# =============================================================================

def binomial_exact(k, n):
    """
    Exact two-sided binomial test against 0.5.

    Used within each culture: are its k source->target propagations (out of n
    propagations in total) more than expected if both directions were equally likely?
    """
    from math import comb
    probabilities = [comb(n, i) / 2 ** n for i in range(n + 1)]
    return min(1.0, sum(q for q in probabilities if q <= probabilities[k] + 1e-12))


def stars(p):
    """Significance label for a figure."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# =============================================================================
# 4. FIGURES
# =============================================================================

def plot_propagations_per_culture(per_culture):
    """
    Number of source->target and target->source propagations in every culture
    (all its recordings summed), with the binomial test of each culture.
    """
    fig, ax = plot_style.new_column_figure()
    table = per_culture.sort_values("n_propagations", ascending=False).reset_index(drop=True)
    x = np.arange(len(table))
    width = 0.38

    # --- Bars ---
    ax.bar(x - width / 2, table["S_to_T"], width, color=plot_style.SOURCE_COLOUR,
           label="Source \u2192 Target")
    ax.bar(x + width / 2, table["T_to_S"], width, color=plot_style.TARGET_COLOUR,
           label="Target \u2192 Source")

    # --- Significance of each culture (binomial test) ---
    for xi, row in table.iterrows():
        top = max(row["S_to_T"], row["T_to_S"])
        label = stars(row["p_binomial"])
        ax.text(xi, top + 2, label, ha="center", va="bottom",
                fontsize=plot_style.ANNOTATION_SIZE if label != "n.s." else plot_style.ANNOTATION_SIZE - 4)

    ax.set_xticks(x)
    ax.set_xticklabels([CULTURE_LABELS[c] for c in table["culture"]])
    ax.set_xlabel("Culture")
    ax.set_xlim(-0.6, len(table) - 0.4)
    ax.set_ylim(0, table["S_to_T"].max() * 1.15)
    ax.set_ylabel("Propagations")
    ax.legend(frameon=False, loc="upper right")
    plot_style.save(fig, HERE, "propagations_per_culture")


def plot_recruitment(per_culture):
    """Recruitment score of the source and of the target chamber, one line per culture."""
    fig, ax = plot_style.new_column_figure()
    x_source, x_target = 0, 1

    # --- One line per culture ---
    for _, row in per_culture.iterrows():
        ax.plot([x_source, x_target], [row["R_source"], row["R_target"]], "-",
                color=plot_style.CULTURE_COLOUR, lw=1.8, zorder=1)
        ax.plot(x_target, row["R_target"], "o", color=plot_style.TARGET_COLOUR, ms=9, zorder=2)
        ax.plot(x_source, row["R_source"], "o", color=plot_style.SOURCE_COLOUR, ms=9, zorder=2)

    # --- Mean over cultures ---
    mean_t, mean_s = per_culture["R_target"].mean(), per_culture["R_source"].mean()
    ax.plot([x_source, x_target], [mean_s, mean_t], "-", color="0.15", lw=3.5, zorder=3)
    ax.plot(x_target, mean_t, "o", color=plot_style.TARGET_COLOUR, ms=17, mec="k", mew=1.5, zorder=4)
    ax.plot(x_source, mean_s, "o", color=plot_style.SOURCE_COLOUR, ms=17, mec="k", mew=1.5, zorder=4)

    ax.set_xticks([x_source, x_target])
    ax.set_xticklabels(["Source", "Target"])
    # --- No significance test: with only 4 mature cultures the figure is descriptive ---
    ax.set_xlim(-0.4, 1.4)
    # --- Period used ---
    ax.text(0.5, 1.08, f"DIV {per_culture.attrs['div_range']}", ha="center", va="bottom",
            fontsize=plot_style.ANNOTATION_SIZE)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Recruitment\nscore")
    plot_style.save(fig, HERE, "recruitment_score")


def plot_time_course(per_stage):
    """Directionality index across maturation stages: individual cultures (grey) and mean +- SEM."""
    fig, ax = plot_style.new_column_figure()
    stages = [stage_of(low) for low, _ in STAGES]
    x = np.arange(len(stages))

    # --- Only cultures with enough propagations in a stage ---
    shown = per_stage[per_stage["n_propagations"] >= MIN_PROPAGATIONS]
    table = shown.pivot(index="culture", columns="stage", values="DI").reindex(columns=stages)

    # --- No preference ---
    ax.axhline(0, color="0.5", ls="--", lw=1.5, zorder=0)

    # --- Individual cultures: grey dots, at the same position as the mean ---
    for xi, stage in zip(x, stages):
        values = table[stage].dropna()
        ax.plot(np.full(len(values), xi), values, "o", color=plot_style.CULTURE_COLOUR, ms=8, zorder=1)

    # --- Mean and standard error over cultures ---
    mean = table.mean(axis=0)
    sem = table.std(axis=0, ddof=1) / np.sqrt(table.count(axis=0))
    ax.errorbar(x, mean, yerr=sem, fmt="-o", color="0.15", lw=3.5, ms=13,
                capsize=6, capthick=2, elinewidth=2, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{low}\u2013{high}" for low, high in STAGES])   # x-axis title says "DIV"
    ax.set_xlim(-0.4, len(stages) - 0.6)
    ax.set_ylim(-1.1, 1.1)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("Days in vitro (DIV)")
    ax.set_ylabel("Directionality\nindex")
    plot_style.save(fig, HERE, "directionality_vs_div")


# =============================================================================
# MAIN
# =============================================================================

def main():
    counts = ["S_to_T", "T_to_S", "local_S", "local_T"]

    # --- Counts per recording ---
    per_recording = add_measures(count_all_recordings())
    per_recording.round(3).to_csv(os.path.join(HERE, "per_recording.csv"), index=False)

    # --- Counts per culture (sum of its recordings) ---
    per_culture = add_measures(per_recording.groupby("culture")[counts].sum().reset_index())
    per_culture["p_binomial"] = [binomial_exact(k, n) for k, n in
                                 zip(per_culture["S_to_T"], per_culture["n_propagations"])]
    per_culture.insert(1, "label", per_culture["culture"].map(CULTURE_LABELS))
    per_culture.round(3).to_csv(os.path.join(HERE, "per_culture.csv"), index=False)

    # --- Counts per culture and maturation stage ---
    per_recording["stage"] = per_recording["div"].map(stage_of)
    per_stage = add_measures(per_recording.groupby(["culture", "stage"])[counts].sum().reset_index())
    per_stage.round(3).to_csv(os.path.join(HERE, "per_stage.csv"), index=False)

    # --- Recruitment score in mature cultures (DIV >= MATURE_FROM_DIV) ---
    mature = per_recording[per_recording["div"] >= MATURE_FROM_DIV]
    per_culture_mature = add_measures(mature.groupby("culture")[counts].sum().reset_index())
    per_culture_mature = per_culture_mature.dropna(subset=["R_source", "R_target"])
    per_culture_mature.round(3).to_csv(os.path.join(HERE, "per_culture_mature.csv"), index=False)
    per_culture_mature.attrs["div_range"] = f"{mature['div'].min()}\u2013{mature['div'].max()}"

    # --- Figures ---
    plot_recruitment(per_culture_mature)
    plot_propagations_per_culture(per_culture)
    plot_time_course(per_stage)

    # --- Results reported in the article ---
    total = per_culture[counts].sum()
    print(per_culture[["label", "culture"] + counts + ["DI", "p_binomial"]].round(3).to_string(index=False))
    print(f"\nAll cultures pooled: {total['S_to_T']} S->T vs {total['T_to_S']} T->S "
          f"({100 * total['S_to_T'] / (total['S_to_T'] + total['T_to_S']):.0f}% source->target)")
    significant = per_culture[(per_culture["p_binomial"] < 0.05) & (per_culture["DI"] > 0)]
    print(f"Cultures with a significant S->T bias (binomial p < 0.05): {len(significant)} of {len(per_culture)} "
          f"(largest p = {significant['p_binomial'].max():.3f})")

    rs, rt = per_culture_mature["R_source"], per_culture_mature["R_target"]
    print(f"\nRecruitment score, DIV {per_culture_mature.attrs['div_range']} ({len(rs)} cultures): "
          f"R(S) = {rs.mean():.2f}, R(T) = {rt.mean():.2f}; R(S) > R(T) in {(rs > rt).sum()} of {len(rs)} cultures")

    print("\nDI per stage (cultures with >= %d propagations):" % MIN_PROPAGATIONS)
    shown = per_stage[per_stage["n_propagations"] >= MIN_PROPAGATIONS]
    order = [stage_of(low) for low, _ in STAGES]
    print(shown.groupby("stage")["DI"].agg(["count", "mean"]).reindex(order).round(2).to_string())


if __name__ == "__main__":
    main()
