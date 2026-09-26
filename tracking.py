"""tracking.py – Simple centroid‑based temporal tracker for KITTI frames.

This module is deliberately lightweight and has no external dependencies beyond NumPy.
It provides:
* load_kitti_timestamps() – cached loading of the optional timestamps file.
* Tracker data structures stored in Streamlit session_state (list of dicts).
* update_tracker() – matches current DBSCAN clusters to existing tracks,
  computes velocity, speed, and status (static / dynamic) using configurable thresholds.

The tracker is *purely* a helper – the main pipeline in app.py remains unchanged.
"""

from __future__ import annotations
import os
import numpy as np
from typing import List, Tuple, Dict, Any
import streamlit as st

# ---------------------------------------------------------------------------
# Cached timestamp loader – returns a list of float timestamps (seconds).
# If the timestamps file does not exist we return an empty list and the caller
# will fall back to a default frame interval.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_kitti_timestamps() -> List[float]:
    """Load KITTI timestamps if present.

    Expected location: ``data/kitti/2011_09_26/2011_09_26_drive_0005_sync/timestamps.txt``
    Each line is a float representing seconds since the start of the sequence.
    Returns an empty list when the file cannot be read.
    """
    ts_path = os.path.join(
        "data",
        "kitti",
        "2011_09_26",
        "2011_09_26_drive_0005_sync",
        "timestamps.txt",
    )
    if not os.path.isfile(ts_path):
        return []
    try:
        with open(ts_path, "r", encoding="utf-8") as f:
            return [float(line.strip()) for line in f if line.strip()]
    except Exception:
        # Any parsing error – treat as missing timestamps.
        return []

# ---------------------------------------------------------------------------
# Helper to compute XY centroid of a point cluster (Nx3 array).
# ---------------------------------------------------------------------------
def _cluster_centroid_xy(cluster: np.ndarray) -> Tuple[float, float, float]:
    """Return (x, y, z) centroid of a cluster.

    The Z component is kept for possible future use, but matching uses only XY.
    """
    # Guard against empty clusters – should never happen, but be safe.
    if cluster.shape[0] == 0:
        return (0.0, 0.0, 0.0)
    c = cluster.mean(axis=0)
    return float(c[0]), float(c[1]), float(c[2])

# ---------------------------------------------------------------------------
# Core tracking routine.
# ---------------------------------------------------------------------------
def update_tracker(
    prev_tracks: List[Dict[str, Any]],
    clusters: List[np.ndarray],
    match_dist: float,
    max_miss: int,
    delta_t: float,
    dyn_speed_thr: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Match current clusters to previous tracks and update all state.

    Parameters
    ----------
    prev_tracks: list of track dicts (from ``st.session_state.tracks``).
    clusters: list of np.ndarray, each cluster's points (shape (M, 3)).
    match_dist: maximum XY distance (meters) to consider a match.
    max_miss: number of consecutive missed frames before a track is removed.
    delta_t: time between the current and previous KITTI frames (seconds).
    dyn_speed_thr: speed (m/s) above which a track is classified as "dynamic".

    Returns
    -------
    (new_tracks, debug_info) where ``debug_info`` contains counts used for UI.
    """
    # Compute centroids for the incoming clusters.
    curr_centroids = [_cluster_centroid_xy(c) for c in clusters]

    # Prepare match structures.
    matched = [False] * len(clusters)
    updated_tracks: List[Dict[str, Any]] = []
    used_prev = set()

    # Greedy matching: iterate over previous tracks and find nearest eligible centroid.
    for track in prev_tracks:
        # If there are no remaining candidates we consider this track missed.
        if not any(not m for m in matched):
            # No candidates left – just increase miss count.
            track["miss_count"] += 1
            if track["miss_count"] <= max_miss:
                updated_tracks.append(track)
            continue

        # Find nearest unmatched centroid.
        best_idx = -1
        best_dist = float("inf")
        tx, ty, _ = track["centroid"]
        for i, (cx, cy, _) in enumerate(curr_centroids):
            if matched[i]:
                continue
            d = np.hypot(cx - tx, cy - ty)
            if d < best_dist:
                best_dist = d
                best_idx = i
        # Accept match if within threshold.
        if best_idx != -1 and best_dist <= match_dist:
            matched[best_idx] = True
            used_prev.add(track["track_id"])
            # Update track with new centroid.
            prev_c = track["centroid"]
            cur_c = np.array(curr_centroids[best_idx])
            # Compute planar velocity.
            if delta_t > 0:
                vx = (cur_c[0] - prev_c[0]) / delta_t
                vy = (cur_c[1] - prev_c[1]) / delta_t
            else:
                vx = vy = 0.0
            speed = float(np.hypot(vx, vy))
            status = "dynamic" if speed >= dyn_speed_thr else "static"
            # Keep a short trail (last 5 centroids).
            trail = track.get("trail", [])
            trail.append((float(cur_c[0]), float(cur_c[1])))
            if len(trail) > 5:
                trail.pop(0)
            updated_tracks.append(
                {
                    "track_id": track["track_id"],
                    "centroid": cur_c,
                    "previous_centroid": prev_c,
                    "velocity": np.array([vx, vy]),
                    "speed": speed,
                    "age": track["age"] + 1,
                    "miss_count": 0,
                    "status": status,
                    "trail": trail,
                }
            )
        else:
            # No match – increase miss count.
            track["miss_count"] += 1
            if track["miss_count"] <= max_miss:
                updated_tracks.append(track)
            # else: drop the track automatically.

    # Create new tracks for any clusters that were not matched.
    next_id = st.session_state.get("next_track_id", 1)
    for i, matched_flag in enumerate(matched):
        if not matched_flag:
            cx, cy, cz = curr_centroids[i]
            # New track has no velocity yet.
            new_track = {
                "track_id": next_id,
                "centroid": np.array([cx, cy, cz]),
                "previous_centroid": np.array([cx, cy, cz]),  # same as current for first frame
                "velocity": np.array([0.0, 0.0]),
                "speed": 0.0,
                "age": 1,
                "miss_count": 0,
                "status": "static",  # will be updated on next frame if it moves
                "trail": [(cx, cy)],
            }
            updated_tracks.append(new_track)
            next_id += 1
    # Store the next available ID back to session_state.
    st.session_state["next_track_id"] = next_id

    # Debug information for UI metrics.
    debug = {
        "new_tracks": len(updated_tracks) - len(prev_tracks) + sum(1 for t in prev_tracks if t["miss_count"] > max_miss),
        "active_tracks": len(updated_tracks),
        "dynamic": sum(1 for t in updated_tracks if t["status"] == "dynamic"),
        "static": sum(1 for t in updated_tracks if t["status"] == "static"),
    }
    return updated_tracks, debug

# ---------------------------------------------------------------------------
# Convenience wrapper to initialise empty tracking state (called once per session).
# ---------------------------------------------------------------------------
def init_tracking_state() -> None:
    if "tracks" not in st.session_state:
        st.session_state["tracks"] = []
    if "next_track_id" not in st.session_state:
        st.session_state["next_track_id"] = 1
