#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topology.py   (STEP 8)
===========

Topological metrics of the effective connectivity networks, reconstructed with
generalized transfer entropy (GTE).

The GTE networks and their metrics were computed with the GTE analysis code of
the laboratory (not included here); this script only reads its output files,
selects the reliable networks, and summarizes the metrics.

    Input : ../software_outputs/GTE/<culture>/DIVxx/     GTE output of every analysed recording
                <recording>_NetMeasures.txt              network metrics for each Z-score threshold
                <recording>_Z2_Matrix_TH.txt             thresholded connectivity matrix (z >= 2)
                <recording>_Z2_node_IDs.txt              ROI index of each row/column of the matrix
            ../5_analysis_data/index.csv                 recordings and source/target chamber
    Output: topology_per_recording.csv                   metrics of every recording, valid or not
            topology_per_culture.csv                     mature metrics, mean of each culture
            topology_cross_chamber.csv                   connections between chambers, per culture
            topology_metrics.png / .pdf                  figure: Q, SW and Geff of mature networks (one dot per culture)

Selection of reliable networks
------------------------------
Graph metrics are unreliable when the network has few active nodes or is broken
into isolated pieces. A recording is kept if
    N_active    >= MIN_ACTIVE_NODES                  (enough active ROIs)
    N_connected >= MIN_CONNECTED_FRACTION * N_active (not fragmented)

Metrics (from NetMeasures, z >= 2)
-------
    Q      modularity of the thresholded network (Q_TH)
    SW     small-worldness index (SmallWorldness_SW)
    Geff   global efficiency of the thresholded network (Geff_TH)

Run:
    python3 topology.py      (requires numpy, pandas and matplotlib)
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
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import recordings   # noqa: E402
import plot_style   # noqa: E402


# =============================================================================
# CONFIGURATION
# =============================================================================

GTE_FOLDER = os.path.join(HERE, "..", "software_outputs", "GTE")
Z_SCORE = 2                       # significance threshold of the GTE connections

MIN_ACTIVE_NODES = 100            # selection of reliable networks (see above)
MIN_CONNECTED_FRACTION = 0.8

MATURE = (14, 24)                 # DIV range of the mature networks (same as the recruitment score)
CHAMBER_SPLIT = 200               # ROIs 1-200 = chamber A, 201-400 = chamber B


# =============================================================================
# 1. READ THE GTE OUTPUT
# =============================================================================

def read_net_measures(path):
    """Row of NetMeasures for the chosen Z-score threshold."""
    data = pd.read_csv(path, skipinitialspace=True)
    data.columns = [c.strip() for c in data.columns]
    row = data[np.isclose(pd.to_numeric(data["Z_SCORE"], errors="coerce"), Z_SCORE)].iloc[0]
    return {
        "N_active": int(row["N_active"]),
        "N_connected": int(row["N_connected"]),
        "Q": float(row["Q_TH"]),
        "SW": float(row["SmallWorldness_SW"]),
        "Geff": float(row["Geff_TH"]),
        "clustering": float(row["Clustering_SW"]),
        "path_length": float(row["ave_l"]),
    }


def cross_chamber_connections(matrix_path, ids_path, source_chamber):
    """
    Number of significant connections from source to target and from target to source.
    The matrix element (i, j) is taken as the connection from node i to node j.
    """
    matrix = np.nan_to_num(np.loadtxt(matrix_path, delimiter=","))
    ids = pd.read_csv(ids_path).iloc[:, 0].to_numpy()
    in_A = ids <= CHAMBER_SPLIT
    A_to_B = int((matrix[np.ix_(in_A, ~in_A)] > 0).sum())
    B_to_A = int((matrix[np.ix_(~in_A, in_A)] > 0).sum())
    return (A_to_B, B_to_A) if source_chamber == "A" else (B_to_A, A_to_B)


def read_all():
    """Metrics of every analysed recording (step 5)."""
    index = pd.read_csv(os.path.join(recordings.ANALYSIS_DATA, "index.csv"))
    rows = []
    for _, rec in index.iterrows():
        folder = os.path.join(GTE_FOLDER, rec["culture"], f"DIV{rec['div']}")
        name = rec["recording"]
        row = {"culture": rec["culture"], "div": rec["div"], "recording": name}

        measures = os.path.join(folder, f"{name}_NetMeasures.txt")
        if not os.path.exists(measures):
            row.update(valid=False, exclusion_reason="no GTE network (too few connections)")
            rows.append(row)
            continue

        row.update(read_net_measures(measures))
        row["S_to_T"], row["T_to_S"] = cross_chamber_connections(
            os.path.join(folder, f"{name}_Z2_Matrix_TH.txt"),
            os.path.join(folder, f"{name}_Z2_node_IDs.txt"),
            rec["source_chamber"])

        # --- Selection of reliable networks ---
        if row["N_active"] < MIN_ACTIVE_NODES:
            reason = f"fewer than {MIN_ACTIVE_NODES} active nodes"
        elif row["N_connected"] < MIN_CONNECTED_FRACTION * row["N_active"]:
            reason = f"fragmented (connected nodes < {MIN_CONNECTED_FRACTION:.0%} of active)"
        else:
            reason = ""
        row.update(valid=reason == "", exclusion_reason=reason)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# 2. STATISTICS
# =============================================================================

def paired_t_test(a, b):
    """
    Two-sided paired Student's t-test (same test as in 6_directionality; implemented
    here so that scipy is not needed). Returns (t, degrees_of_freedom, p_value).
    """
    from math import gamma, sqrt, pi
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    df = n - 1
    t = d.mean() / (d.std(ddof=1) / sqrt(n))
    # P(-|t| < T < |t|) by integrating the t density from 0 to |t| (Simpson's rule)
    x = np.linspace(0, abs(t), 20001)
    density = (gamma((df + 1) / 2) / (sqrt(df * pi) * gamma(df / 2))
               * (1 + x ** 2 / df) ** (-(df + 1) / 2))
    h = x[1] - x[0]
    integral = h / 3 * (density[0] + density[-1] + 4 * density[1:-1:2].sum() + 2 * density[2:-1:2].sum())
    return float(t), df, float(1 - 2 * integral)


# =============================================================================
# 3. FIGURE
# =============================================================================

def plot_metrics(mature):
    """Q, SW and Geff of the mature networks: one dot per culture (mean of its recordings), box = quartiles."""
    plot_style.apply()
    # Single-letter variables in italics (Q, G), multi-letter names and labels upright (SW, eff)
    panels = [("Q", "Modularity", "$Q$"), ("SW", "Small-worldness", "SW"),
              ("Geff", "Global efficiency", "$G_{\\mathrm{eff}}$")]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 6), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, (column, title, label) in zip(axes, panels):
        values = mature[column].dropna().to_numpy()
        ax.boxplot(values, widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#dfe6ee", edgecolor="#34495e"),
                   medianprops=dict(color="#e74c3c", lw=2),
                   whiskerprops=dict(color="#34495e"), capprops=dict(color="#34495e"))
        ax.scatter(1 + rng.normal(0, 0.05, len(values)), values, s=130, color="#2c6fbb",
                   alpha=0.6, edgecolor="white", lw=0.6, zorder=3)
        ax.set_title(title, fontsize=plot_style.AXIS_LABEL_SIZE + 2, fontweight="bold", pad=12)
        ax.set_ylabel(label, fontsize=plot_style.AXIS_LABEL_SIZE + 8)
        ax.set_xticks([])
        ax.tick_params(labelsize=plot_style.TICK_LABEL_SIZE + 6)
    fig.suptitle(f"DIV {MATURE[0]}\u2013{MATURE[1]}", fontsize=plot_style.AXIS_LABEL_SIZE + 8)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"topology_metrics.{ext}"), dpi=300)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    table = read_all()
    table.round(3).to_csv(os.path.join(HERE, "topology_per_recording.csv"), index=False)

    valid = table[table["valid"]]
    print(f"Recordings: {len(table)}, valid networks: {len(valid)}")
    print(table.loc[~table["valid"], ["recording", "exclusion_reason"]].to_string(index=False))

    # --- Metrics of mature networks ---
    mature = valid[(valid["div"] >= MATURE[0]) & (valid["div"] <= MATURE[1])]
    # The culture is the statistical unit: each culture is the mean of its recordings.
    metrics = ["Q", "SW", "Geff", "clustering", "path_length"]
    per_culture = mature.groupby("culture")[metrics].mean().reset_index()
    per_culture.insert(1, "label", per_culture["culture"].map(recordings.CULTURE_LABELS))
    per_culture["n_recordings"] = mature.groupby("culture").size().to_numpy()
    per_culture.round(3).to_csv(os.path.join(HERE, "topology_per_culture.csv"), index=False)

    print(f"\nMature networks (DIV {MATURE[0]}-{MATURE[1]}), one value per culture "
          f"({len(per_culture)} cultures, {len(mature)} recordings):")
    print(per_culture.round(2).to_string(index=False))
    for column in metrics:
        v = per_culture[column]
        print(f"  {column:12s} mean = {v.mean():.2f} +- {v.std(ddof=1):.2f}   "
              f"(cultures with value > 1: {(v > 1).sum()})" if column == "SW" else
              f"  {column:12s} mean = {v.mean():.2f} +- {v.std(ddof=1):.2f}")
    plot_metrics(per_culture)

    # --- Connections between chambers in each direction (per culture, ALL valid networks) ---
    # The question (is there an asymmetry in the GTE connections?) does not depend on
    # maturation, so all valid recordings are used: the 7 cultures enter the test.
    cross = valid.groupby("culture")[["S_to_T", "T_to_S"]].sum().reset_index()
    cross.insert(1, "label", cross["culture"].map(recordings.CULTURE_LABELS))
    cross.insert(2, "n_recordings", valid.groupby("culture").size().to_numpy())
    cross["DI_topo"] = (cross["S_to_T"] - cross["T_to_S"]) / (cross["S_to_T"] + cross["T_to_S"])
    cross.round(3).to_csv(os.path.join(HERE, "topology_cross_chamber.csv"), index=False)
    t, df, p_t = paired_t_test(cross["S_to_T"], cross["T_to_S"])
    print("\nGTE connections between chambers (all valid networks, summed per culture):")
    print(cross.round(2).to_string(index=False))
    print(f"S->T vs T->S across cultures: two-sided paired t-test t({df}) = {t:.2f}, p = {p_t:.3f}")


if __name__ == "__main__":
    main()
