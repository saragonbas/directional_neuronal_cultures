# -*- coding: utf-8 -*-
"""
bursts.py
=========

Shared functions used by every analysis in this project.

    1. load_spikes()           Read a spikesData.csv file.
    2. detect_chamber_bursts() Detect network bursts separately in each chamber.
       chamber_activity(), burst_segments()  Same, keeping the activity signal and
                                 the start/end of each burst (used for figures).
    3. match_bursts()          Pair bursts of one chamber with bursts of the other.
    4. chance_matches()        Number of pairs expected by chance (surrogate data).

Chamber definition
------------------
The 400 ROIs are split by index into the two chambers of the device:
    chamber A = ROIs 1-200
    chamber B = ROIs 201-400
"""

import numpy as np
import pandas as pd


# =============================================================================
# PARAMETERS
# =============================================================================

# ROIs with index <= CHAMBER_SPLIT belong to chamber A, the rest to chamber B.
CHAMBER_SPLIT = 200

# A neuron is considered "active" if it fires at least this many spikes.
MIN_SPIKES_ACTIVE = 2

# Time resolution of the activity signal (s).
BIN_SIZE = 0.10

# A neuron counts as co-active at time t if it fires within this window around t (s).
COACTIVITY_WINDOW = 0.50

# A chamber is "bursting" when at least this fraction of its active neurons is co-active.
BURST_THRESHOLD = 0.10

# Two above-threshold segments closer than this (s) are merged into a single burst.
MERGE_GAP = 0.30

# Width (s) of the Gaussian used to smooth the activity signal.
SMOOTHING_WIDTH = 0.10


# =============================================================================
# 1. LOADING DATA
# =============================================================================

def load_spikes(path):
    """
    Read a NETCAL spikesData.csv file.

    Returns
    -------
    neuron_ids : array of int    ROI index of each spike.
    times      : array of float  Time of each spike (s).
    """
    data = pd.read_csv(path, skipinitialspace=True)
    data.columns = [c.strip().lower() for c in data.columns]

    id_column = [c for c in data.columns if "neuron" in c or "index" in c][0]
    time_column = [c for c in data.columns if "time" in c][0]

    neuron_ids = data[id_column].to_numpy(int)
    times = data[time_column].to_numpy(float)
    return neuron_ids, times


def active_neurons_per_chamber(neuron_ids):
    """
    Return the list of active neurons of chamber A and of chamber B.
    A neuron is active if it fires at least MIN_SPIKES_ACTIVE spikes.
    """
    ids, spike_counts = np.unique(neuron_ids, return_counts=True)
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


def _chamber_activity(neuron_ids, times, active_neurons, n_bins):
    """
    Activity signal of one chamber: fraction of its active neurons that fire
    within COACTIVITY_WINDOW around each time bin (smoothed).
    """
    n_neurons = len(active_neurons)
    if n_neurons == 0:
        return np.zeros(n_bins)

    # --- Binary matrix: neuron x time bin (1 if the neuron fires in that bin) ---
    row_of_neuron = {neuron: row for row, neuron in enumerate(active_neurons)}
    is_active_spike = np.array([n in row_of_neuron for n in neuron_ids])

    rows = [row_of_neuron[n] for n in neuron_ids[is_active_spike]]
    cols = np.clip((times[is_active_spike] / BIN_SIZE).astype(int), 0, n_bins - 1)

    fires = np.zeros((n_neurons, n_bins), dtype=np.float32)
    fires[rows, cols] = 1.0

    # --- Does each neuron fire within the co-activity window around each bin? ---
    half_window = int(round((COACTIVITY_WINDOW / BIN_SIZE) / 2))
    cumulative = np.cumsum(fires, axis=1)
    first = np.clip(np.arange(n_bins) - half_window - 1, -1, n_bins - 1)
    last = np.clip(np.arange(n_bins) + half_window, 0, n_bins - 1)
    spikes_in_window = cumulative[:, last] - np.where(first >= 0, cumulative[:, first], 0.0)

    # --- Fraction of co-active neurons, smoothed ---
    fraction_coactive = (spikes_in_window > 0).sum(axis=0) / n_neurons
    return _gaussian_smooth(fraction_coactive, SMOOTHING_WIDTH / BIN_SIZE)


def burst_segments(activity, bin_edges):
    """
    Detect bursts in an activity signal.

    A burst is a segment where activity >= BURST_THRESHOLD. Segments separated
    by less than MERGE_GAP are merged. Each burst is represented by the time at
    which its activity is maximal (peak).

    Returns a list of [start (s), end (s), peak time (s)], one per burst.
    """
    above = np.where(activity >= BURST_THRESHOLD)[0]
    if len(above) == 0:
        return []

    # --- Split the above-threshold bins into contiguous segments ---
    segments = np.split(above, np.where(np.diff(above) > 1)[0] + 1)
    bursts = []
    for seg in segments:
        start = bin_edges[seg[0]]
        end = bin_edges[seg[-1] + 1]
        peak_bin = seg[np.argmax(activity[seg])]
        bursts.append([start, end, peak_bin])

    # --- Merge segments that are too close to each other ---
    merged = [bursts[0]]
    for start, end, peak_bin in bursts[1:]:
        previous = merged[-1]
        if start - previous[1] <= MERGE_GAP:
            previous[1] = end
            if activity[peak_bin] > activity[previous[2]]:
                previous[2] = peak_bin
        else:
            merged.append([start, end, peak_bin])

    # --- Peak time = centre of the peak bin ---
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return [[float(start), float(end), float(bin_centres[peak_bin])] for start, end, peak_bin in merged]


def _burst_peaks(activity, bin_edges):
    """Peak time (s) of each burst of an activity signal."""
    return [peak for _, _, peak in burst_segments(activity, bin_edges)]


def chamber_activity(path):
    """
    Activity signal (chamber-specific GNA) of both chambers of one recording.

    Returns a dictionary with:
        neuron_ids, times   : the spikes of the recording
        bin_edges, centres  : time bins (s)
        activity_A/B        : activity signal of chamber A and B
    """
    neuron_ids, times = load_spikes(path)
    active_A, active_B = active_neurons_per_chamber(neuron_ids)
    duration = float(times.max())
    bin_edges = np.arange(0, duration + BIN_SIZE, BIN_SIZE)
    n_bins = len(bin_edges) - 1
    return {
        "neuron_ids": neuron_ids,
        "times": times,
        "bin_edges": bin_edges,
        "centres": 0.5 * (bin_edges[:-1] + bin_edges[1:]),
        "activity_A": _chamber_activity(neuron_ids, times, active_A, n_bins),
        "activity_B": _chamber_activity(neuron_ids, times, active_B, n_bins),
        "active_A": active_A,
        "active_B": active_B,
    }


def detect_chamber_bursts(path):
    """
    Detect the bursts of each chamber in one recording.

    Returns a dictionary with:
        bursts_A, bursts_B : burst peak times (s) in chamber A and B
        duration           : recording duration (s)
        active_A, active_B : number of active neurons in each chamber
    """
    neuron_ids, times = load_spikes(path)
    active_A, active_B = active_neurons_per_chamber(neuron_ids)

    duration = float(times.max())
    bin_edges = np.arange(0, duration + BIN_SIZE, BIN_SIZE)
    n_bins = len(bin_edges) - 1

    activity_A = _chamber_activity(neuron_ids, times, active_A, n_bins)
    activity_B = _chamber_activity(neuron_ids, times, active_B, n_bins)

    return {
        "bursts_A": _burst_peaks(activity_A, bin_edges),
        "bursts_B": _burst_peaks(activity_B, bin_edges),
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
    propagated the activity to the other.

    Returns
    -------
    A_to_B  : pairs in which A peaked first
    B_to_A  : pairs in which B peaked first
    local_A : bursts of A without a partner in B
    local_B : bursts of B without a partner in A
    """
    used_B = set()
    A_to_B = B_to_A = local_A = 0

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
            if t_A <= bursts_B[best]:
                A_to_B += 1
            else:
                B_to_A += 1

    local_B = len(bursts_B) - len(used_B)
    return A_to_B, B_to_A, local_A, local_B


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
        A_to_B, B_to_A, _, _ = match_bursts(bursts_A, shifted_B, window)
        matches[k] = A_to_B + B_to_A

    return matches
