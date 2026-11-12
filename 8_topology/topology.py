#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topology.py   (STEP 8)
===========

Topology of the effective connectivity networks, reconstructed with generalized
transfer entropy (GTE).

The GTE networks were computed with the analysis code of the laboratory (not included
here). This script reads them, keeps the reliable ones and computes everything shown
in Fig. 6c, Table ST2 and the text.

    Input : ../software_outputs/GTE/<culture>/DIVxx/     GTE results of every recording
                <recording>_NetMeasures.txt              metrics computed by the GTE code
                <recording>_Z2_Matrix_TH.txt             significant connections (z >= 2)
                <recording>_Z2_node_IDs.txt              ROI of each row/column of the matrix
            ../5_analysis_data/index.csv                 source and target chamber of each culture
    Output: topology_per_recording.csv                   every recording, valid or not
            topology_null_and_chambers.csv               mature recordings: random networks and chambers
            topology_per_culture.csv                     mature recordings, mean of each culture
            topology_cross_chamber.csv                   Table ST2: connections between chambers and reciprocity
            topology_metrics.png / .pdf                  Fig. 6c

1. Reliable networks
   Graph metrics are not reliable when a network has few active ROIs or is broken into
   pieces. A recording is kept if it has at least MIN_ACTIVE_NODES active ROIs and at
   least MIN_CONNECTED_FRACTION of them have some connection.

2. Metrics (Fig. 6c), computed by the GTE code
   Q (modularity), SW (small-worldness) and Geff (global efficiency).

3. Comparison with random networks (Fig. 6c)
   Each network is compared with N_RANDOM random networks, made by shuffling its
   connections while every ROI keeps its number of incoming and outgoing connections.
   Q and Geff of the random networks are computed in the same way as in the GTE code
   (the script checks that it gives the same values for the real networks).
   SW does not need this, because it is already relative to a random network (SW = 1).

4. Communities and chambers (text)
   f_cross        fraction of connections between the two chambers, compared with
                  f_cross_rand, the fraction expected if the connections were placed
                  at random among the active ROIs of the recording
   Q_chambers     modularity when the two chambers are taken as the two groups
   NMI            similarity between the communities and the two chambers
                  (0 = unrelated, 1 = identical)
   n_communities  number of communities with at least MIN_COMMUNITY_SIZE ROIs
   same_chamber   fraction of the ROIs of each community that are in the same chamber

5. Direction of the connections (Table ST2)
   Connections from source to target and from target to source, compared with a paired
   t-test across cultures, and reciprocity: fraction of connections i->j for which j->i
   also exists, compared with the value expected by chance (the connection density).

Each culture is represented by the mean of its recordings.

Run:
    python3 topology.py      (requires numpy, pandas, matplotlib and networkx; takes ~1 min)
"""

import os
import sys

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import recordings
import plot_style


# =============================================================================
# CONFIGURATION
# =============================================================================

GTE_FOLDER = os.path.join(HERE, "..", "software_outputs", "GTE")
Z_SCORE = 2                       # significance threshold of the GTE connections

MIN_ACTIVE_NODES = 100            # selection of reliable networks (see above)
MIN_CONNECTED_FRACTION = 0.8

MATURE = (14, 24)                 # DIV range of the mature networks (same as the recruitment score)

N_RANDOM = 20                     # random networks per real network
SWAPS_PER_CONNECTION = 5                # rewiring steps per connection in each random network
LOUVAIN_RUNS_REAL = 5             # Louvain runs (best partition) for the real networks
LOUVAIN_RUNS_RANDOM = 2           # Louvain runs (best partition) for each random network
MIN_COMMUNITY_SIZE = 5            # communities counted in n_communities
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
        "path_length": float(row["ave_L_topo"]),   # mean shortest path length (ave_l is the physical length of the connections)
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


def reciprocity(matrix_path):
    """
    Fraction of connections i->j for which j->i also exists, and the value expected by
    chance in a random network: the connection density, number of connections / N(N-1).
    """
    matrix = np.nan_to_num(np.loadtxt(matrix_path, delimiter=",")) > 0
    np.fill_diagonal(matrix, False)
    n = len(matrix)
    return {"reciprocity": (matrix & matrix.T).sum() / matrix.sum(),
            "reciprocity_chance": matrix.sum() / (n * (n - 1))}


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
        row.update(reciprocity(os.path.join(folder, f"{name}_Z2_Matrix_TH.txt")))

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
    Two-sided paired Student's t-test (implemented here so that scipy is not needed).
    Returns (t, degrees_of_freedom, p_value).
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
# 3. RANDOM NETWORKS, COMMUNITIES AND CHAMBERS
# =============================================================================

def load_network(prefix):
    """Directed, unweighted network from the thresholded GTE matrix, and the ROI index of each node."""
    matrix = np.nan_to_num(np.loadtxt(prefix + "_Z2_Matrix_TH.txt", delimiter=",")) > 0
    np.fill_diagonal(matrix, False)
    ids = pd.read_csv(prefix + "_Z2_node_IDs.txt").iloc[:, 0].to_numpy()
    return nx.from_numpy_array(matrix.astype(int), create_using=nx.DiGraph), ids


def best_partition(graph, runs):
    """Louvain communities with the highest modularity over several runs."""
    return max((louvain_communities(graph, seed=seed) for seed in range(runs)),
               key=lambda communities: modularity(graph, communities))


def global_efficiency(graph):
    """
    Mean of 1/d_ij over all pairs of ROIs, where d_ij is the number of steps of the
    shortest path from i to j following the direction of the connections (0 if there
    is no path). Distances are found step by step: first the ROIs reached in one step,
    then in two, and so on.
    """
    connections = nx.to_numpy_array(graph, dtype=np.float32) > 0
    n = len(connections)
    reached = np.eye(n, dtype=bool)
    frontier = reached.copy()
    total, distance = 0.0, 0
    while frontier.any():
        distance += 1
        frontier = ((frontier.astype(np.float32) @ connections.astype(np.float32)) > 0) & ~reached
        total += frontier.sum() / distance
        reached |= frontier
    return total / (n * (n - 1))


def random_network(graph, seed):
    """
    Random network made by shuffling the connections: two connections a->b and c->d are
    swapped to a->d and c->b, many times. Every ROI keeps its number of incoming and
    outgoing connections.
    """
    rewired = graph.copy()
    n_connections = rewired.number_of_edges()
    nx.directed_edge_swap(rewired, nswap=SWAPS_PER_CONNECTION * n_connections,
                          max_tries=200 * n_connections, seed=seed)
    return rewired


def normalized_mutual_information(a, b):
    """NMI between two partitions of the same nodes (Danon et al. 2005)."""
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    joint = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(joint, (a, b), 1)
    joint /= joint.sum()
    pa, pb = joint.sum(1), joint.sum(0)
    nonzero = joint > 0
    mutual = (joint[nonzero] * np.log(joint[nonzero] / np.outer(pa, pb)[nonzero])).sum()
    entropy = lambda p: -(p[p > 0] * np.log(p[p > 0])).sum()
    return 2 * mutual / (entropy(pa) + entropy(pb))


def null_and_chambers(prefix):
    """Q and Geff of the random networks, and comparison of the communities with the chambers."""
    graph, ids = load_network(prefix)
    row = {}

    # --- Random networks ---
    q_random, geff_random = [], []
    for seed in range(N_RANDOM):
        rewired = random_network(graph, seed)
        q_random.append(max(modularity(rewired, louvain_communities(rewired, seed=s))
                            for s in range(LOUVAIN_RUNS_RANDOM)))
        geff_random.append(global_efficiency(rewired))
    row["Q_random"], row["Geff_random"] = np.mean(q_random), np.mean(geff_random)

    # --- Communities vs chambers ---
    communities = best_partition(graph, LOUVAIN_RUNS_REAL)
    chamber = {n: int(ids[n] > CHAMBER_SPLIT) for n in graph}
    connected = [n for n in graph if graph.degree(n) > 0]
    community = {n: i for i, c in enumerate(communities) for n in c}
    chamber_partition = [c for c in ({n for n in graph if chamber[n] == k} for k in (0, 1)) if c]
    row["Q_python"] = modularity(graph, communities)          # check: same as Q of the GTE code
    row["Geff_python"] = global_efficiency(graph)              # check: same as Geff of the GTE code
    row["Q_chambers"] = modularity(graph, chamber_partition)
    n_a = sum(chamber[n] == 0 for n in graph)                  # active ROIs of each chamber
    n_b = graph.number_of_nodes() - n_a
    n = n_a + n_b
    between = sum(chamber[i] != chamber[j] for i, j in graph.edges())
    row["f_cross"] = between / graph.number_of_edges()
    row["f_cross_rand"] = 2 * n_a * n_b / (n * (n - 1))
    row["NMI"] = normalized_mutual_information([community[n] for n in connected],
                                               [chamber[n] for n in connected])
    members = [[n for n in c if n in set(connected)] for c in communities]
    row["n_communities"] = sum(len(m) >= MIN_COMMUNITY_SIZE for m in members)
    row["same_chamber"] = sum(max(sum(chamber[n] == 0 for n in m), sum(chamber[n] == 1 for n in m))
                              for m in members) / len(connected)
    return row


# =============================================================================
# 4. FIGURE
# =============================================================================

def plot_metrics(per_culture):
    """Q, SW and Geff of the mature networks (one dot per culture, mean of its recordings),
    compared with random networks. Boxes show the median and the quartiles."""
    plot_style.apply()
    real_colour, random_colour = "#2c6fbb", "#8a8f98"
    panels = [("Q", "Modularity", "$Q$", 0.8), ("SW", "Small-worldness", "SW", 6),
              ("Geff", "Global efficiency", "$G_{\\mathrm{eff}}$", 0.5)]
    shift = np.linspace(-0.09, 0.09, len(per_culture))

    def box(ax, position, values, face, edge):
        ax.boxplot(values, positions=[position], widths=0.42, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=face, edgecolor=edge, lw=1.3),
                   medianprops=dict(color="#e74c3c", lw=2.2),
                   whiskerprops=dict(color=edge, lw=1.3), capprops=dict(color=edge, lw=1.3))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 7), constrained_layout=True)
    for ax, (column, title, label, top) in zip(axes, panels):
        real = per_culture[column].to_numpy()
        if column == "SW":
            # --- SW is already relative to a random network (SW = 1) ---
            ax.axhspan(0, 1, color=random_colour, alpha=0.12, lw=0)
            ax.axhline(1, color=random_colour, ls="--", lw=1.6)
            ax.text(0.5, 1.12, "Random", color=random_colour, ha="center", va="bottom",
                    fontsize=plot_style.TICK_LABEL_SIZE + 6)
            box(ax, 0.5, real, "#dfe6ee", "#34495e")
            ax.scatter(0.5 + shift, real, s=170, color=real_colour, alpha=0.8, edgecolor="white", zorder=3)
            ax.set_xticks([0.5])
            ax.set_xticklabels(["Real"])
            ax.set_xlim(0, 1)
        else:
            random = per_culture[column + "_random"].to_numpy()
            for i in range(len(real)):
                ax.plot([shift[i], 1 + shift[i]], [real[i], random[i]], color="0.82", lw=1.2, zorder=1)
            box(ax, 0, real, "#dfe6ee", "#34495e")
            box(ax, 1, random, "white", random_colour)
            ax.scatter(shift, real, s=170, color=real_colour, alpha=0.8, edgecolor="white", zorder=3)
            ax.scatter(1 + shift, random, s=170, facecolor="white", edgecolor=random_colour, lw=1.8, zorder=3)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Real", "Random"])
            ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, top)
        ax.set_title(title, fontsize=plot_style.AXIS_LABEL_SIZE + 6, fontweight="bold", pad=14)
        ax.set_ylabel(label, fontsize=plot_style.AXIS_LABEL_SIZE + 16)
        ax.tick_params(axis="x", length=0, labelsize=plot_style.TICK_LABEL_SIZE + 10)
        ax.tick_params(axis="y", labelsize=plot_style.TICK_LABEL_SIZE + 10)
    fig.suptitle(f"DIV {MATURE[0]}\u2013{MATURE[1]}", fontsize=plot_style.AXIS_LABEL_SIZE + 14)
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

    # --- Random networks and partition defined by the two chambers, for every mature network ---
    extra = pd.DataFrame([null_and_chambers(os.path.join(GTE_FOLDER, rec["culture"], f"DIV{rec['div']}",
                                                         rec["recording"]))
                          for _, rec in mature.iterrows()], index=mature.index)
    mature = pd.concat([mature, extra], axis=1)
    mature[["culture", "div", "recording", "Q", "Q_python", "Q_random", "Geff", "Geff_python", "Geff_random",
            "f_cross", "f_cross_rand", "Q_chambers", "NMI", "n_communities", "same_chamber"]].round(4).to_csv(
        os.path.join(HERE, "topology_null_and_chambers.csv"), index=False)
    extra_metrics = ["Q_random", "Geff_random", "f_cross", "f_cross_rand", "Q_chambers", "NMI",
                     "n_communities", "same_chamber"]
    per_culture = per_culture.merge(mature.groupby("culture")[extra_metrics].mean().reset_index(), on="culture")
    per_culture.round(3).to_csv(os.path.join(HERE, "topology_per_culture.csv"), index=False)

    print(f"\nMature networks (DIV {MATURE[0]}-{MATURE[1]}), one value per culture "
          f"({len(per_culture)} cultures, {len(mature)} recordings):")
    print(per_culture.round(2).to_string(index=False))
    lower = (mature["f_cross"] < mature["f_cross_rand"]).sum()
    print(f"  f_cross below its random expectation in {lower} of {len(mature)} recordings")
    print(f"  (check: Q and Geff recomputed here differ from the GTE code by at most "
          f"{(mature['Q_python'] - mature['Q']).abs().max():.3f} and "
          f"{(mature['Geff_python'] - mature['Geff']).abs().max():.3f}; GTE values are rounded to 0.01)")
    for column in metrics + extra_metrics:
        v = per_culture[column]
        print(f"  {column:14s} mean +- SD = {v.mean():.2f} +- {v.std(ddof=1):.2f}"
              + (f"   (cultures with value > 1: {(v > 1).sum()})" if column == "SW" else ""))
    plot_metrics(per_culture)

    # --- Connections between chambers in each direction (per culture, all valid networks) ---
    # The question (is there an asymmetry in the GTE connections?) does not depend on
    # maturation, so all valid recordings are used: the 7 cultures enter the test.
    cross = valid.groupby("culture")[["S_to_T", "T_to_S"]].sum().reset_index()
    cross.insert(1, "label", cross["culture"].map(recordings.CULTURE_LABELS))
    cross.insert(2, "n_recordings", valid.groupby("culture").size().to_numpy())
    cross["DI_topo"] = (cross["S_to_T"] - cross["T_to_S"]) / (cross["S_to_T"] + cross["T_to_S"])
    # Reciprocity: mean of the valid recordings of each culture
    recip = ["reciprocity", "reciprocity_chance"]
    cross = cross.merge(valid.groupby("culture")[recip].mean().reset_index(), on="culture")
    cross.round(3).to_csv(os.path.join(HERE, "topology_cross_chamber.csv"), index=False)
    t, df, p_t = paired_t_test(cross["S_to_T"], cross["T_to_S"])
    print("\nGTE connections between chambers (all valid networks, summed per culture):")
    print(cross.round(2).to_string(index=False))
    print(f"S->T vs T->S across cultures: two-sided paired t-test t({df}) = {t:.2f}, p = {p_t:.3f}")
    print("Reciprocity (fraction of connections i->j with j->i also present; mean +- SD across cultures):")
    names = {"reciprocity": "reciprocity r", "reciprocity_chance": "density rho (= r expected by chance)"}
    for column in recip:
        print(f"  {names[column]:37s} {cross[column].mean():.2f} +- {cross[column].std(ddof=1):.2f}   "
              f"(range {cross[column].min():.2f}-{cross[column].max():.2f})")


if __name__ == "__main__":
    main()
