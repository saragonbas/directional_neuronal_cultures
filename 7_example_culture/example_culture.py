#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_culture.py   (STEP 7)
==================

Figures of the example culture (C1 = 15_05_1, DIV 17), made with exactly
the same burst detection and matching (Delta t = 1 s) as the rest of the analysis.

    Input : ../5_analysis_data/15_05_1/DIV17/      spikes and ROI positions
    Output: example_raster.png / .pdf              (a) raster + chamber GNA, events coloured by type
            example_zoom.png / .pdf                (b) zoom on one source->target propagation
            example_state_diagram.png / .pdf       (c) number of events of each type
            example_initiation.png / .pdf          (d) initiation site of every burst
            example_events.csv                     every burst of the recording, with its type

Event types (see 6_directionality)
-----------
    S->T   burst of the source matched to a later burst of the target
    T->S   burst of the target matched to a later burst of the source
    S->0   local burst of the source (no partner in the target), labelled "S local"
    T->0   local burst of the target (no partner in the source), labelled "T local"

Run:
    python3 example_culture.py      (requires numpy, pandas and matplotlib)
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.colors import ListedColormap

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import bursts
import recordings
import plot_style


# =============================================================================
# CONFIGURATION
# =============================================================================

CULTURE = "15_05_1"          # culture C1 in the article
DIV = 17
SOURCE_CHAMBER = "A"         # ROIs 1-200 are the source in this culture
COINCIDENCE_WINDOW = 1.0     # Delta t (s), chosen in step 3

RASTER_TMAX = 200            # (a) seconds shown in the raster
PANEL_HEIGHT = 8.5           # (a) and (b) have the same height (inches) to be placed side by side
ZOOM = (305.8, 310.8)        # (b) time window of the zoom (s)

# (d) physical size of the imaged field (mm) and limits of the chambers (mm, y axis)
FIELD_X, FIELD_Y = 4.7, 2.7
SOURCE_Y_MIN, TARGET_Y_MAX = 1.5, 1.2
# Initiation site, as in VisualizeFront: among the first N_CANDIDATES ROIs to fire, the group
# of GROUP_SIZE consecutive ROIs closest to each other; spikes beyond OUTLIER_SD standard
# deviations of the mean burst time are removed first (twice).
N_CANDIDATES, GROUP_SIZE, OUTLIER_SD = 10, 4, 2
REPETITION_WINDOW = 0.5      # repeated spikes of one ROI within this window (s) are removed first

# Colours of the event types
COLOURS = {"S->T": plot_style.SOURCE_COLOUR, "T->S": plot_style.TARGET_COLOUR,
           "S->0": "#41b6c4", "T->0": "#fdae61", "S=T": "#9e9e9e"}
LABELS = {"S->T": "S→T", "T->S": "T→S", "S->0": "S local", "T->0": "T local", "S=T": "Simultaneous"}

FOLDER = os.path.join(recordings.ANALYSIS_DATA, CULTURE, f"DIV{DIV}")
NAME = f"{CULTURE}_DIV{DIV}"


# =============================================================================
# 1. BURSTS AND THEIR CLASSIFICATION
# =============================================================================

def classify_events(source_bursts, target_bursts, window):
    """
    Same matching as bursts.match_bursts, but keeping every event.

    Each burst of the source is paired with the closest unused burst of the
    target within `window`; the chamber that peaks first is the origin. If both
    peak in the same time step, the event is simultaneous ('S=T') and both
    chambers are taken as its origin.

    Returns a list of dicts: type, start, end, origin chamber ('S' or 'T'),
    and the start/end of the burst in the origin chamber.
    """
    used_T = set()
    events = []
    for s_start, s_end, s_peak in sorted(source_bursts, key=lambda b: b[2]):
        best, best_distance = None, window
        for j, (_, _, t_peak) in enumerate(target_bursts):
            distance = abs(t_peak - s_peak)
            if j not in used_T and distance <= best_distance:
                best, best_distance = j, distance

        if best is None:
            events.append(dict(type="S->0", start=s_start, end=s_end, origin="S",
                               origin_start=s_start, origin_end=s_end, peak=s_peak))
            continue

        used_T.add(best)
        t_start, t_end, t_peak = target_bursts[best]
        if abs(s_peak - t_peak) < 1e-9:
            events.append(dict(type="S=T", start=min(s_start, t_start), end=max(s_end, t_end),
                               origin="S+T", origin_start=min(s_start, t_start),
                               origin_end=max(s_end, t_end), peak=s_peak))
        elif s_peak < t_peak:
            events.append(dict(type="S->T", start=min(s_start, t_start), end=max(s_end, t_end),
                               origin="S", origin_start=s_start, origin_end=s_end, peak=s_peak))
        else:
            events.append(dict(type="T->S", start=min(s_start, t_start), end=max(s_end, t_end),
                               origin="T", origin_start=t_start, origin_end=t_end, peak=t_peak))

    for j, (t_start, t_end, t_peak) in enumerate(target_bursts):
        if j not in used_T:
            events.append(dict(type="T->0", start=t_start, end=t_end, origin="T",
                               origin_start=t_start, origin_end=t_end, peak=t_peak))

    return sorted(events, key=lambda e: e["start"])


def load_example():
    """Chamber GNA, bursts and events of the example recording."""
    data = bursts.chamber_gna(os.path.join(FOLDER, f"{NAME}_spikesData.csv"))
    edges = data["bin_edges"]

    # --- Source / target instead of chamber A / B ---
    s, t = ("A", "B") if SOURCE_CHAMBER == "A" else ("B", "A")
    data["gna_S"], data["gna_T"] = data[f"gna_{s}"], data[f"gna_{t}"]
    data["active_S"], data["active_T"] = data[f"active_{s}"], data[f"active_{t}"]
    data["source_is_low"] = SOURCE_CHAMBER == "A"   # source = ROIs 1-200

    source_bursts = bursts.burst_segments(data["gna_S"], edges)
    target_bursts = bursts.burst_segments(data["gna_T"], edges)
    events = classify_events(source_bursts, target_bursts, COINCIDENCE_WINDOW)

    # --- Check: same counts as the main analysis (bursts.match_bursts) ---
    counts = bursts.match_bursts([b[2] for b in source_bursts], [b[2] for b in target_bursts],
                                 COINCIDENCE_WINDOW)
    mine = tuple(sum(e["type"] == k for e in events) for k in ("S->T", "T->S", "S->0", "T->0", "S=T"))
    assert mine == counts, (mine, counts)

    return data, events


def is_source_roi(roi, data):
    return roi <= bursts.CHAMBER_SPLIT if data["source_is_low"] else roi > bursts.CHAMBER_SPLIT


# =============================================================================
# 2. FIGURES
# =============================================================================

def save(fig, name, tight=True):
    # tight=False keeps the exact figure size (used for panels that must have the same height)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=300, bbox_inches="tight" if tight else None)
    plt.close(fig)


def plot_raster(data, events):
    """(a) Raster of the first RASTER_TMAX s, events shaded by type, and chamber GNA."""
    plot_style.apply()
    fig, (ax_r, ax_g) = plt.subplots(2, 1, figsize=(10, PANEL_HEIGHT), height_ratios=[3.1, 1.2],
                                     sharex=True, constrained_layout=True)
    ids, times = data["roi_ids"], data["times"]

    # --- Events (shaded, slightly widened so that short events are visible) ---
    # Propagations cover both chambers; local bursts only the ROIs of their own chamber.
    split = bursts.CHAMBER_SPLIT + 0.5
    low_is_origin = {"S": data["source_is_low"], "T": not data["source_is_low"]}
    for e in events:
        if e["start"] > RASTER_TMAX:
            continue
        centre, width = (e["start"] + e["end"]) / 2, (e["end"] - e["start"]) + 1.0
        x = [centre - width / 2, centre + width / 2]
        if e["type"] in ("S->0", "T->0"):
            y0, y1 = (0, split) if low_is_origin[e["origin"]] else (split, 400)
        else:
            y0, y1 = 0, 400
        ax_r.fill_between(x, y0, y1, color=COLOURS[e["type"]], alpha=0.6, lw=0)

    # --- Raster ---
    m = times <= RASTER_TMAX
    ax_r.scatter(times[m], ids[m], s=2.4, c="0.05", marker="|", linewidths=1)
    ax_r.axhline(bursts.CHAMBER_SPLIT + 0.5, color="k", lw=1.4, ls="--")
    ax_r.set_ylim(400, 0)
    ax_r.set_ylabel("ROI", fontsize=plot_style.AXIS_LABEL_SIZE + 3)
    ax_r.tick_params(labelsize=plot_style.TICK_LABEL_SIZE + 2)
    ax_r.legend(handles=[Patch(facecolor=COLOURS[k], label=LABELS[k]) for k in COLOURS],
                loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=False,
                fontsize=plot_style.LEGEND_SIZE + 4, handlelength=1.2, columnspacing=1.2, handletextpad=0.5)

    # --- Chamber GNA ---
    c = data["centres"]
    ax_g.plot(c, data["gna_S"], color=plot_style.SOURCE_COLOUR, lw=1.2)
    ax_g.plot(c, data["gna_T"], color=plot_style.TARGET_COLOUR, lw=1.2)
    ax_g.axhline(bursts.BURST_THRESHOLD, color="0.5", ls=":", lw=1.0)
    ax_g.set_xlim(0, RASTER_TMAX)
    ax_g.set_ylim(0, 1)
    ax_g.set_ylabel("GNA", fontsize=plot_style.AXIS_LABEL_SIZE + 3)
    ax_g.set_xlabel("Time (s)", fontsize=plot_style.AXIS_LABEL_SIZE + 3)
    ax_g.tick_params(labelsize=plot_style.TICK_LABEL_SIZE + 2)
    save(fig, "example_raster", tight=False)


def plot_zoom(data, events):
    """(b) Zoom on one source->target propagation, with the delay between GNA peaks."""
    plot_style.apply()
    t0, t1 = ZOOM
    fig, (ax_r, ax_g) = plt.subplots(2, 1, figsize=(4.3, PANEL_HEIGHT), height_ratios=[3.1, 1.2],
                                     sharex=True, constrained_layout=True)
    ids, times = data["roi_ids"], data["times"]
    m = (times >= t0) & (times <= t1)
    src = np.array([is_source_roi(i, data) for i in ids])

    # --- Raster: source blue, target red ---
    ax_r.scatter(times[m & src] - t0, ids[m & src], s=30, c=plot_style.SOURCE_COLOUR,
                 marker="|", linewidths=1.6)
    ax_r.scatter(times[m & ~src] - t0, ids[m & ~src], s=30, c=plot_style.TARGET_COLOUR,
                 marker="|", linewidths=1.6)
    ax_r.axhline(bursts.CHAMBER_SPLIT + 0.5, color="k", lw=1.4, ls="--")
    ax_r.set_ylim(400, 0)
    ax_r.set_yticks(range(0, 401, 100))
    ax_r.set_ylabel("ROI", fontsize=plot_style.AXIS_LABEL_SIZE + 4)
    ax_r.tick_params(labelsize=plot_style.TICK_LABEL_SIZE + 2)
    # Legend above the raster, as in (a), so that the axes of both panels are aligned
    ax_r.legend(handles=[Patch(facecolor=plot_style.SOURCE_COLOUR, label="Source"),
                         Patch(facecolor=plot_style.TARGET_COLOUR, label="Target")],
                loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False,
                fontsize=plot_style.LEGEND_SIZE + 3, handlelength=1.5, columnspacing=1.6)

    # --- GNA of each chamber and delay between peaks ---
    c = data["centres"]
    g = (c >= t0) & (c <= t1)
    ax_g.plot(c[g] - t0, data["gna_S"][g], color=plot_style.SOURCE_COLOUR, lw=2.6)
    ax_g.plot(c[g] - t0, data["gna_T"][g], color=plot_style.TARGET_COLOUR, lw=2.6)
    s_peak = c[g][np.argmax(data["gna_S"][g])] - t0
    t_peak = c[g][np.argmax(data["gna_T"][g])] - t0
    y_arrow = 0.95
    for x, col in ((s_peak, plot_style.SOURCE_COLOUR), (t_peak, plot_style.TARGET_COLOUR)):
        ax_g.plot([x, x], [0, y_arrow], color=col, ls=":", lw=2)
    ax_g.annotate("", xy=(t_peak, y_arrow), xytext=(s_peak, y_arrow),
                  arrowprops=dict(arrowstyle="->", color="k", lw=1.2, shrinkA=0, shrinkB=0))
    ax_g.text((s_peak + t_peak) / 2, y_arrow + 0.07, f"delay \u2248 {t_peak - s_peak:.1f} s",
              ha="center", va="bottom", fontsize=plot_style.ANNOTATION_SIZE)
    ax_g.set_xlim(0, t1 - t0)
    ax_g.set_ylim(0, 1.3)
    ax_g.set_yticks([0, 0.5, 1.0])
    ax_g.set_ylabel("GNA", fontsize=plot_style.AXIS_LABEL_SIZE + 4)
    ax_g.set_xlabel("Time (s)", fontsize=plot_style.AXIS_LABEL_SIZE + 4)
    ax_g.tick_params(labelsize=plot_style.TICK_LABEL_SIZE + 2)
    save(fig, "example_zoom", tight=False)

    # --- Check that the zoomed burst is a S->T propagation ---
    kind = [e["type"] for e in events if e["start"] <= t0 + s_peak + 0.5 and e["end"] >= t0 + s_peak - 0.5]
    return t_peak - s_peak, kind


def plot_state_diagram(events):
    """(c) Burst state diagram: arrow width proportional to the number of events.
    Local bursts (S->0, T->0) are drawn as a loop that leaves and re-enters their own chamber;
    simultaneous events (S=T) as a grey line without direction."""
    plot_style.apply()
    n = {k: sum(e["type"] == k for e in events) for k in COLOURS}
    n_max = max(n.values())
    x_range, y_range, inches_per_unit = 2.7, 2.6, 1.5      # nearly square panel
    fig = plt.figure(figsize=(x_range * inches_per_unit, y_range * inches_per_unit))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(-x_range / 2, x_range / 2)
    ax.set_ylim(-y_range / 2, y_range / 2)
    ax.set_aspect("equal")
    points_per_unit = 72 * inches_per_unit
    pos = {"S": np.array([0.0, 0.8]), "T": np.array([0.0, -0.8])}
    node_colour = {"S": plot_style.SOURCE_COLOUR, "T": plot_style.TARGET_COLOUR}
    radius = 0.34

    def width(count):
        return 1.2 + 5.0 * count / n_max

    def label(xy, count, colour):
        ax.text(*xy, str(count), color=colour, fontsize=plot_style.ANNOTATION_SIZE, weight="bold",
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="none"))

    def arrow(a, b, count, colour, rad):
        pa, pb = pos[a], pos[b]
        ax.add_patch(FancyArrowPatch(pa, pb, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                     mutation_scale=15, lw=width(count), color=colour,
                                     shrinkA=radius * points_per_unit, shrinkB=radius * points_per_unit,
                                     alpha=0.9, zorder=2))
        d = pb - pa
        perp = np.array([d[1], -d[0]]) / np.hypot(*d)
        label((pa + pb) / 2 + perp * rad * np.hypot(*d) * 0.6, count, colour)

    def loop(a, count, colour, direction):
        # Cubic Bezier curve leaving the circle and coming back to it on the outer side
        centre, reach = pos[a], 0.8
        angles = np.radians([direction + 35, direction - 35])
        unit = [np.array([np.cos(t), np.sin(t)]) for t in angles]
        p1, p2 = centre + radius * unit[0], centre + radius * unit[1]
        c1, c2 = p1 + reach * unit[0], p2 + reach * unit[1]
        path = Path([p1, c1, c2, p2], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=15, lw=width(count),
                                     color=colour, alpha=0.9, zorder=2))
        label((p1 + 3 * c1 + 3 * c2 + p2) / 8, count, colour)

    if n["S=T"]:
        ax.plot([0, 0], [pos["S"][1] - radius, pos["T"][1] + radius], color=COLOURS["S=T"],
                lw=width(n["S=T"]), ls=(0, (2, 1.5)), alpha=0.9, zorder=1, solid_capstyle="butt")
        label((0, 0), n["S=T"], "0.35")
    arrow("S", "T", n["S->T"], COLOURS["S->T"], -0.30)
    arrow("T", "S", n["T->S"], COLOURS["T->S"], -0.30)
    loop("S", n["S->0"], COLOURS["S->0"], 180)
    loop("T", n["T->0"], COLOURS["T->0"], 0)
    for k, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), radius, fc=node_colour[k], ec="white", lw=2, zorder=3))
        ax.text(x, y, k, ha="center", va="center", color="white", fontsize=plot_style.ANNOTATION_SIZE + 2,
                weight="bold", zorder=4)
    save(fig, "example_state_diagram")
    return n


def remove_repetitions(ids, times, window):
    """Keep only the first spike of each ROI within `window` s (as in VisualizeFront)."""
    keep = np.zeros(len(times), bool)
    last_id, last_time = None, -np.inf
    for k in np.lexsort((times, ids)):          # sorted by ROI, then by time
        if ids[k] != last_id or times[k] - last_time > window:
            keep[k] = True
            last_id, last_time = ids[k], times[k]
    return ids[keep], times[keep]


def plot_initiation(data, events):
    """(d) Initiation site of every burst, computed as in VisualizeFront (see N_CANDIDATES)."""
    plot_style.apply()
    roi = pd.read_csv(os.path.join(FOLDER, f"{NAME}_ROI_centers_everything.csv"), skipinitialspace=True)
    roi.columns = [c.strip().lower() for c in roi.columns]
    id_col = [c for c in roi.columns if "id" in c][0]
    x_mm = (roi.x - roi.x.min()) / (roi.x.max() - roi.x.min()) * FIELD_X
    y_mm = (roi.y - roi.y.min()) / (roi.y.max() - roi.y.min()) * FIELD_Y
    xy = {int(i): (x, y) for i, x, y in zip(roi[id_col], x_mm, y_mm)}

    ids, times = remove_repetitions(data["roi_ids"], data["times"], REPETITION_WINDOW)
    points = []
    for e in events:
        m = (times >= e["start"]) & (times <= e["end"])
        burst_ids, burst_times = ids[m], times[m]

        # --- Remove spikes far from the mean time of the burst (twice) ---
        for _ in range(2):
            if len(burst_times) > 2:
                keep = np.abs(burst_times - burst_times.mean()) <= OUTLIER_SD * burst_times.std()
                burst_ids, burst_times = burst_ids[keep], burst_times[keep]

        # --- ROIs of both chambers ranked by their first spike in the burst ---
        first = {}
        for i, t in zip(burst_ids, burst_times):
            if i not in first or t < first[i]:
                first[i] = t
        ordered = [i for i in sorted(first, key=first.get) if i in xy]
        if not ordered:
            continue
        pos = np.array([xy[i] for i in ordered])

        # --- Most compact group of GROUP_SIZE consecutive ROIs among the first ones ---
        n = min(N_CANDIDATES, len(pos) - GROUP_SIZE + 1)
        if n < 1:
            group = pos
        else:
            spread = [np.mean([np.hypot(*(pos[k + a] - pos[k + b]))
                               for a in range(GROUP_SIZE) for b in range(a + 1, GROUP_SIZE)])
                      for k in range(n)]
            k = int(np.argmin(spread))
            group = pos[k:k + GROUP_SIZE]
        points.append((group[:, 0].mean(), group[:, 1].mean(), e["type"]))

    # Chamber where each burst starts: source, channels or target
    def region_of(y):
        return "source" if y > SOURCE_Y_MIN else "target" if y < TARGET_Y_MAX else "channels"
    initiation = pd.crosstab(pd.Series([LABELS[k] for _, _, k in points], name="burst type"),
                             pd.Series([region_of(y) for _, y, _ in points], name="initiation site"))

    fig, ax = plt.subplots(figsize=(8, 5))
    # --- Background: source (blue), channels (grey), target (red) ---
    n_grid = 20
    xe, ye = np.linspace(0, FIELD_X, n_grid + 1), np.linspace(0, FIELD_Y, n_grid + 1)
    yc = (np.arange(n_grid) + 0.5) / n_grid * FIELD_Y
    region = np.repeat(np.where(yc > SOURCE_Y_MIN, 2, np.where(yc < TARGET_Y_MAX, 0, 1))[:, None], n_grid, 1)
    ax.pcolormesh(xe, ye, region, cmap=ListedColormap(["#fbe3e3", "#e6e6e6", "#dfeaf7"]),
                  vmin=0, vmax=2, edgecolors="white", linewidth=0.6, zorder=0)
    # --- Density of initiation sites (Gaussian kernel) ---
    h = 0.22
    gx, gy = np.meshgrid(np.linspace(0, FIELD_X, 160), np.linspace(0, FIELD_Y, 160))
    density = sum(np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * h * h)) for x, y, _ in points)
    density /= density.max()
    ax.contourf(gx, gy, density, levels=np.linspace(0.05, 1, 9), cmap="Reds", alpha=0.9, zorder=1)
    ax.contour(gx, gy, density, levels=np.linspace(0.15, 1, 6), colors="#7a0000",
               linewidths=0.7, alpha=0.6, zorder=1)
    # --- One dot per burst ---
    for x, y, kind in points:
        ax.scatter(x, y, color=COLOURS[kind], s=55, edgecolor="white", lw=0.8, zorder=3, clip_on=False)
    # Chamber names next to the channels
    ax.text(FIELD_X - 0.1, SOURCE_Y_MIN + 0.04, "SOURCE", color=plot_style.SOURCE_COLOUR, fontsize=16,
            fontweight="bold", ha="right", va="bottom")
    ax.text(FIELD_X - 0.1, TARGET_Y_MAX - 0.04, "TARGET", color=plot_style.TARGET_COLOUR, fontsize=16,
            fontweight="bold", ha="right", va="top")
    ax.set_xlim(0, FIELD_X)
    ax.set_ylim(0, FIELD_Y)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOURS[k],
                              markersize=11, label=LABELS[k]) for k in COLOURS],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, frameon=False, fontsize=15,
              handletextpad=0.2, columnspacing=0.9)   # below the map, so that no point is hidden
    save(fig, "example_initiation")
    return initiation


# =============================================================================
# MAIN
# =============================================================================

def main():
    data, events = load_example()
    pd.DataFrame(events).round(2).to_csv(os.path.join(HERE, "example_events.csv"), index=False)

    plot_raster(data, events)
    delay, zoom_type = plot_zoom(data, events)
    counts = plot_state_diagram(events)
    initiation = plot_initiation(data, events)

    print(f"{NAME} (C1), Delta t = {COINCIDENCE_WINDOW} s")
    print("Events:", counts)
    print(f"Zoom: event type {zoom_type}, delay between GNA peaks = {delay:.2f} s")
    led_by_source = counts["S->T"] + counts["S->0"]
    print(f"Bursts led by the source that recruited the target: {counts['S->T']} of {led_by_source}")
    print("Initiation site of each burst (number of bursts):")
    print(initiation.to_string())


if __name__ == "__main__":
    main()
