# -*- coding: utf-8 -*-
"""
bursts.py
=========

Shared functions used by every analysis in this project.

    1. load_spikes()           Read a spikesData.csv file.
    2. detect_chamber_bursts() Detect the chamber bursts: network bursts detected
                               separately in each chamber from its chamber GNA.
       chamber_gna(), burst_segments()  Same, keeping the chamber GNA and the
                                 start/end of each burst (used for figures).
    3. match_bursts()          Pair bursts of one chamber with bursts of the other.
    4. chance_matches()        Number of pairs expected by chance (surrogate data).

Chamber definition
------------------
The 400 ROIs are split by index into the two chambers of the device:
    chamber A = top chamber    (ROIs 1-200)
    chamber B = bottom chamber (ROIs 201-400)
Which of them is the source depends on the culture (see 4_culture_selection).

Chamber GNA
-----------
As the GNA of the whole culture, but computed within each chamber: fraction of
the active ROIs of the chamber that fire within a 0.5 s window centred on each
time step. A chamber burst is an episode in which the chamber GNA exceeds 0.1.
"""

import numpy as np
import pandas as pd


# =============================================================================
# PARAMETERS
# =============================================================================

# ROIs with index <= CHAMBER_SPLIT belong to chamber A, the rest to chamber B.
CHAMBER_SPLIT = 200

# A ROI is considered "active" if it fires at least this many spikes.
MIN_SPIKES_ACTIVE = 2

# Time step of the chamber GNA (s).
BIN_SIZE = 0.10

# Window (s) of the chamber GNA: a ROI counts as active at time t if it fires within
# this window centred on t.
GNA_WINDOW = 0.50

# A chamber burst is detected when the chamber GNA exceeds this value
# (fraction of the active ROIs of the chamber).
BURST_THRESHOLD = 0.10

# Two above-threshold segments closer than this (s) are merged into a single burst.
MERGE_GAP = 0.30

# Width (s) of the Gaussian used to smooth the chamber GNA. Without smoothing, the
# chamber GNA changes in small steps and its maximum can fall one step earlier or
# later by chance; smoothing places the peak at the centre of the burst, which is
# needed to decide which chamber peaked first.
SMOOTHING_WIDTH = 0.10


# =============================================================================
# 1. LOADING DATA
# =============================================================================

def load_spikes(path):
    """
    Read a spikesData.csv file (exported from Netcal).

    Returns
    -------
    roi_ids    : array of int    ROI index of each spike.
    times      : array of float  Time of each spike (s).
    """
    data = pd.read_csv(path, skipinitialspace=True)
    data.columns = [c.strip().lower() for c in data.columns]

    # Netcal names the ROI column "neuron" (or "index")
    id_column = [c for c in data.columns if "neuron" in c or "index" in c][0]
    time_column = [c for c in data.columns if "time" in c][0]

    roi_ids = data[id_column].to_numpy(int)
    times = data[time_column].to_numpy(float)
    return roi_ids, times


def active_rois_per_chamber(roi_ids):
    """
    Return the list of active ROIs of chamber A and of chamber B.
    A ROI is active if it fires at least MIN_SPIKES_ACTIVE spikes.
    """
    ids, spike_counts = np.unique(roi_ids, return_counts=True)
    active = ids[spike_counts >= MIN_SPIKES_ACTIVE]

    active_A = [n for n in active if n <= CHAMBER_SPLIT]
    active_B = [n for n in active if n > CHAMBER_SPLIT]
    return active_A, active_B


# =============================================================================
# 2. BURST DETECTION IN EACH CHAMBER
# =============================================================================

def _gaussian_smooth(signal, width_in_bins):
    """Smooth a 1-D signal with a normalised Gaussian kernel."""
    half_length = max(1, int(round(3 * width_in_bins)))
    x = np.arange(-half_length, half_length + 1)
    kernel = np.exp(-x ** 2 / (2 * width_in_bins ** 2))
    kernel /= kernel.sum()
    return np.convolve(signal, kernel, mode="same")


def _chamber_gna(roi_ids, times, active_rois, n_bins):
    """
    Chamber GNA: fraction of the active ROIs of one chamber that fire within
    GNA_WINDOW around each time step (smoothed).
    """
    n_rois = len(active_rois)
    if n_rois == 0:
        return np.zeros(n_bins)

    # --- Binary matrix: ROI x time step (1 if the ROI fires in that step) ---
    row_of_roi = {roi: row for row, roi in enumerate(active_rois)}
    is_active_spike = np.array([n in row_of_roi for n in roi_ids])

    rows = [row_of_roi[n] for n in roi_ids[is_active_spike]]
    cols = np.clip((times[is_active_spike] / BIN_SIZE).astype(int), 0, n_bins - 1)

    fires = np.zeros((n_rois, n_bins), dtype=np.float32)
    fires[rows, cols] = 1.0

    # --- Does each ROI fire within the window centred on each time step? ---
    half_window = int(round((GNA_WINDOW / BIN_SIZE) / 2))
    cumulative = np.cumsum(fires, axis=1)
    first = np.clip(np.arange(n_bins) - half_window - 1, -1, n_bins - 1)
    last = np.clip(np.arange(n_bins) + half_window, 0, n_bins - 1)
    spikes_in_window = cumulative[:, last] - np.where(first >= 0, cumulative[:, first], 0.0)

    # --- Fraction of active ROIs that fire in the window, smoothed ---
    gna = (spikes_in_window > 0).sum(axis=0) / n_rois
    return _gaussian_smooth(gna, SMOOTHING_WIDTH / BIN_SIZE)


def burst_segments(gna, bin_edges):
    """
    Detect the chamber bursts in a chamber GNA.

    A burst is a segment where the chamber GNA >= BURST_THRESHOLD. Segments separated
    by less than MERGE_GAP are merged. Each burst is represented by the time at
    which its chamber GNA is maximal (peak).

    Returns a list of [start (s), end (s), peak time (s)], one per burst.
    """
    above = np.where(gna >= BURST_THRESHOLD)[0]
    if len(above) == 0:
        return []

    # --- Split the above-threshold bins into contiguous segments ---
    segments = np.split(above, np.where(np.diff(above) > 1)[0] + 1)
    bursts = []
    for seg in segments:
        start = bin_edges[seg[0]]
        end = bin_edges[seg[-1] + 1]
        peak_bin = seg[np.argmax(gna[seg])]
        bursts.append([start, end, peak_bin])

    # --- Merge segments that are too close to each other ---
    merged = [bursts[0]]
    for start, end, peak_bin in bursts[1:]:
        previous = merged[-1]
        if start - previous[1] <= MERGE_GAP:
            previous[1] = end
            if gna[peak_bin] > gna[previous[2]]:
                previous[2] = peak_bin
        else:
            merged.append([start, end, peak_bin])

    # --- Peak time = centre of the peak bin ---
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return [[float(start), float(end), float(bin_centres[peak_bin])] for start, end, peak_bin in merged]


def _burst_peaks(gna, bin_edges):
    """Peak time (s) of each burst of a chamber GNA."""
    return [peak for _, _, peak in burst_segments(gna, bin_edges)]


def chamber_gna(path):
    """
    Chamber GNA of both chambers of one recording.

    Returns a dictionary with:
        roi_ids, times      : the spikes of the recording
        bin_edges, centres  : time steps (s)
        gna_A, gna_B        : chamber GNA of chamber A and B
    """
    roi_ids, times = load_spikes(path)
    active_A, active_B = active_rois_per_chamber(roi_ids)
    duration = float(times.max())
    bin_edges = np.arange(0, duration + BIN_SIZE, BIN_SIZE)
    n_bins = len(bin_edges) - 1
    return {
        "roi_ids": roi_ids,
        "times": times,
        "bin_edges": bin_edges,
        "centres": 0.5 * (bin_edges[:-1] + bin_edges[1:]),
        "gna_A": _chamber_gna(roi_ids, times, active_A, n_bins),
        "gna_B": _chamber_gna(roi_ids, times, active_B, n_bins),
        "active_A": active_A,
        "active_B": active_B,
    }


def detect_chamber_bursts(path):
    """
    Detect the chamber bursts of both chambers in one recording.

    Returns a dictionary with:
        bursts_A, bursts_B : burst peak times (s) in chamber A and B
        duration           : recording duration (s)
        active_A, active_B : number of active ROIs in each chamber
    """
    roi_ids, times = load_spikes(path)
    active_A, active_B = active_rois_per_chamber(roi_ids)

    duration = float(times.max())
    bin_edges = np.arange(0, duration + BIN_SIZE, BIN_SIZE)
    n_bins = len(bin_edges) - 1

    gna_A = _chamber_gna(roi_ids, times, active_A, n_bins)
    gna_B = _chamber_gna(roi_ids, times, active_B, n_bins)

    return {
        "bursts_A": _burst_peaks(gna_A, bin_edges),
        "bursts_B": _burst_peaks(gna_B, bin_edges),
        "duration": duration,
        "active_A": len(active_A),
        "active_B": len(active_B),
    }


# =============================================================================
# 3. MATCHING BURSTS BETWEEN CHAMBERS
# =============================================================================

def match_bursts(bursts_A, bursts_B, window):
    """
    Pair each burst of chamber A with the closest unused burst of chamber B,
    provided their peaks are at most `window` seconds apart.

    In each pair, the chamber whose burst peaks first is the one that
    propagated the activity to the other. If both peaks fall in the same time
    step, the order cannot be decided and the pair is counted as simultaneous
    (a coincident burst without direction).

    Returns
    -------
    A_to_B  : pairs in which A peaked first
    B_to_A  : pairs in which B peaked first
    local_A : bursts of A without a partner in B
    local_B : bursts of B without a partner in A
    simultaneous : pairs in which both chambers peaked in the same time step
    """
    used_B = set()
    A_to_B = B_to_A = local_A = simultaneous = 0

    for t_A in sorted(bursts_A):
        # --- Find the closest burst of B not yet used, within the window ---
        best, best_distance = None, window
        for j, t_B in enumerate(bursts_B):
            distance = abs(t_B - t_A)
            if j not in used_B and distance <= best_distance:
                best, best_distance = j, distance

        # --- Classify the burst of A ---
        if best is None:
            local_A += 1
        else:
            used_B.add(best)
            if abs(t_A - bursts_B[best]) < 1e-9:
                simultaneous += 1
            elif t_A < bursts_B[best]:
                A_to_B += 1
            else:
                B_to_A += 1

    local_B = len(bursts_B) - len(used_B)
    return A_to_B, B_to_A, local_A, local_B, simultaneous


# =============================================================================
# 4. MATCHES EXPECTED BY CHANCE (SURROGATE DATA)
# =============================================================================

def chance_matches(bursts_A, bursts_B, duration, window, n_repetitions, rng):
    """
    Number of matched pairs expected by chance.

    All burst times of chamber B are shifted by the same random offset; bursts
    shifted beyond the end of the recording wrap around to its beginning
    (circular shift). This keeps the number and the temporal pattern of bursts
    in each chamber, but removes any timing relationship between chambers.

    Returns an array with the number of matches in each repetition.
    """
    if len(bursts_A) == 0 or len(bursts_B) == 0:
        return np.zeros(n_repetitions)

    bursts_B = np.asarray(bursts_B)
    matches = np.empty(n_repetitions)

    for k in range(n_repetitions):
        offset = rng.uniform(5, duration - 5)
        shifted_B = np.sort((bursts_B + offset) % duration).tolist()
        A_to_B, B_to_A, _, _, simultaneous = match_bursts(bursts_A, shifted_B, window)
        matches[k] = A_to_B + B_to_A + simultaneous

    return matches
