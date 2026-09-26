"""Autonomous Perception & Variable-Resolution Grid Dashboard.

DRDO Problem Statement PS26053:
Real-time LiDAR perception pipeline with adaptive spatial resolution.
Supports both Synthetic Simulation and Real KITTI Velodyne HDL-64E datasets.
"""

import os
import glob
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Existing core modules
from tracking import load_kitti_timestamps, init_tracking_state, update_tracker
import pandas as pd
from data_gen import generate_frame
from terrain_segmentation import segment_ground
from object_clustering import cluster_objects, classify_static_vs_dynamic
from variable_resolution_grid import build_variable_grid, memory_savings, TIERS
from kitti_loader import load_lidar_frame

# ---------------------------------------------------------
# Page Configuration & Technical Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Autonomous Perception | Variable-Resolution LiDAR Grid",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Technical CSS for Aerospace / Defense Console Appearance
st.markdown(
    """
    <style>
    /* Global Background & Font */
    .stApp {
        background-color: #070a12;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Header Container */
    .telemetry-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 10px;
        margin-bottom: 8px;
    }
    .telemetry-title {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        color: #f8fafc;
        text-transform: uppercase;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .telemetry-subtitle {
        font-size: 0.95rem;
        font-weight: 600;
        color: #38bdf8;
        letter-spacing: 0.02em;
    }
    .telemetry-tagline {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 2px;
    }
    .status-badge-online {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        color: #34d399;
        letter-spacing: 0.08em;
        font-family: ui-monospace, monospace;
    }
    .status-dot-online {
        width: 7px;
        height: 7px;
        background-color: #10b981;
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 8px #10b981;
    }
    .status-badge-error {
        display: inline-flex;
        align-items: center;
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        color: #f87171;
        letter-spacing: 0.08em;
        font-family: ui-monospace, monospace;
    }
    .pipeline-breadcrumb {
        font-size: 0.7rem;
        color: #475569;
        font-family: ui-monospace, monospace;
        letter-spacing: 0.04em;
        margin-bottom: 12px;
    }
    
    /* Primary Telemetry Cards */
    .metric-card {
        background: #0c1222;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .metric-card.highlight {
        background: #091a18;
        border: 1px solid #059669;
    }
    .metric-label {
        font-size: 0.68rem;
        font-weight: 700;
        color: #94a3b8;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-family: ui-monospace, monospace;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #f8fafc;
        font-family: ui-monospace, monospace;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .metric-value.green {
        color: #10b981;
    }
    .metric-value.cyan {
        color: #38bdf8;
    }
    
    /* Secondary Diagnostics Strip */
    .diag-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        background: #080d1a;
        border: 1px solid #162032;
        border-radius: 4px;
        padding: 6px 12px;
        margin-bottom: 14px;
        font-family: ui-monospace, monospace;
        font-size: 0.72rem;
        color: #64748b;
    }
    .diag-item {
        display: flex;
        gap: 6px;
    }
    .diag-item span.val {
        color: #cbd5e1;
        font-weight: 600;
    }
    
    /* Custom Legend Bar */
    .legend-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        background: #090e1a;
        border: 1px solid #1e293b;
        border-radius: 4px;
        padding: 6px 12px;
        margin-top: 6px;
        margin-bottom: 12px;
        font-size: 0.74rem;
        font-family: ui-monospace, monospace;
        color: #94a3b8;
        align-items: center;
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    
    /* Collapsible & Technical Sections */
    .streamlit-expanderHeader {
        background-color: #0c1222 !important;
        border: 1px solid #1e293b !important;
        color: #cbd5e1 !important;
        font-family: ui-monospace, monospace !important;
        font-size: 0.8rem !important;
    }
    
    /* Sidebar Tightening */
    section[data-testid="stSidebar"] {
        background-color: #080c16;
        border-right: 1px solid #162032;
    }
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #94a3b8;
        font-family: ui-monospace, monospace;
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Session State Initialization (Extended for Tracking)
# ---------------------------------------------------------
if "pedestrian_offset" not in st.session_state:
    st.session_state.pedestrian_offset = 1.2
if "synthetic_frame_id" not in st.session_state:
    st.session_state.synthetic_frame_id = 2
if "centroid_history" not in st.session_state:
    st.session_state.centroid_history = [(20.0, 5.0)]
if "kitti_frame_idx" not in st.session_state:
    st.session_state.kitti_frame_idx = 0
# Tracking state
if "tracks" not in st.session_state:
    st.session_state.tracks = []
if "next_track_id" not in st.session_state:
    st.session_state.next_track_id = 1
# (prev_data_source init removed)

# Timestamp loading moved after data_source definition

# ---------------------------------------------------------
# Reset handling (clear tracking state)
# ---------------------------------------------------------
if data_source == "Synthetic":
    reset_button = st.sidebar.button("Reset 🔄", key="reset_syn")
    if reset_button:
        st.session_state.pedestrian_offset = 1.2
        st.session_state.synthetic_frame_id = 2
        st.session_state.centroid_history = [(20.0, 5.0)]
        st.session_state.tracks = []
        st.session_state.next_track_id = 1
else:
    reset_button = st.sidebar.button("Reset 🔄", key="reset_kitti")
    if reset_button:
        st.session_state.kitti_frame_idx = 0
        st.session_state.tracks = []
        st.session_state.next_track_id = 1

# Detect data source change and reset tracking
if st.session_state.prev_data_source != data_source:
    st.session_state.prev_data_source = data_source
    st.session_state.tracks = []
    st.session_state.next_track_id = 1

# ---------------------------------------------------------
# Expanded Perception Parameters (add tracking sliders)
# ---------------------------------------------------------
with st.sidebar.expander("PERCEPTION", expanded=False):
    default_ground_thresh = 0.08 if data_source == "Synthetic" else 0.15
    ground_thresh = st.slider(
        "Ground RANSAC Threshold (m)",
        min_value=0.02,
        max_value=0.30,
        value=default_ground_thresh,
        step=0.01,
        help="Max orthogonal distance to fitted ground plane.",
    )

    default_dbscan_eps = 0.60 if data_source == "Synthetic" else 0.80
    dbscan_eps = st.slider(
        "DBSCAN Epsilon (m)",
        min_value=0.20,
        max_value=1.50,
        value=default_dbscan_eps,
        step=0.05,
        help="Spatial clustering radius.",
    )

    default_dbscan_min_pts = 15 if data_source == "Synthetic" else 25
    dbscan_min_pts = st.slider(
        "DBSCAN Min Points",
        min_value=5,
        max_value=50,
        value=default_dbscan_min_pts,
        step=1,
        help="Minimum core points per cluster.",
    )

    max_range = st.slider(
        "Max Planar Range (m)",
        min_value=20.0,
        max_value=120.0,
        value=100.0,
        step=5.0,
        help="Filter points beyond planar distance sqrt(x² + y²).",
    )

    # --- Tracking specific sliders ---
    track_match_dist = st.slider(
        "TRACK_MATCH_DISTANCE (m)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="Maximum XY distance to match a cluster to an existing track.",
    )
    dyn_speed_thr = st.slider(
        "DYNAMIC_SPEED_THRESHOLD (m/s)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Speed above which a track is considered dynamic.",
    )
    max_miss = st.slider(
        "MAX_MISS (frames)",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Frames a track can be missed before removal.",
    )

# ---------------------------------------------------------
# Perception Pipeline Execution (add tracking for KITTI)
# ---------------------------------------------------------
try:
    t_start = time.time()

    if data_source == "Synthetic":
        # existing synthetic pipeline unchanged ... (omitted for brevity)
        pass
    else:  # KITTI Real LiDAR
        if not kitti_files:
            raise RuntimeError("KITTI dataset files not found. Check path: data/kitti/...")

        kitti_file_path = kitti_files[st.session_state.kitti_frame_idx]
        pts_raw, _intensity = load_lidar_frame(kitti_file_path)
        valid_mask = np.isfinite(pts_raw).all(axis=1)
        pts_live = pts_raw[valid_mask]
        if max_range > 0:
            r = np.hypot(pts_live[:, 0], pts_live[:, 1])
            pts_live = pts_live[r <= max_range]
        total_raw_points = len(pts_live)
        ground_mask = segment_ground(pts_live, distance_threshold=ground_thresh)
        ground_points_count = int(np.sum(ground_mask))
        non_ground_points_count = total_raw_points - ground_points_count
        clusters = cluster_objects(pts_live, ~ground_mask, eps=dbscan_eps, min_points=dbscan_min_pts)
        num_obstacle_clusters = len(clusters)
        ground_pts = pts_live[ground_mask]
        ground_lbls = np.full(len(ground_pts), "ground", dtype=object)
        cluster_pts_list = []
        cluster_lbl_list = []
        for c in clusters:
            cluster_pts_list.append(c)
            cluster_lbl_list.append(np.full(len(c), "static_obstacle", dtype=object))
        if cluster_pts_list:
            final_points = np.vstack([ground_pts] + cluster_pts_list)
            final_labels = np.concatenate([ground_lbls] + cluster_lbl_list)
        else:
            final_points = pts_live
            final_labels = np.full(len(final_points), "ground" if ground_points_count > 0 else "unknown", dtype=object)

        # ---- Tracking Update ----
        # Determine delta_t using timestamps if available
        if st.session_state.kitti_timestamps and st.session_state.kitti_frame_idx > 0:
            delta_t = (
                st.session_state.kitti_timestamps[st.session_state.kitti_frame_idx]
                - st.session_state.kitti_timestamps[st.session_state.kitti_frame_idx - 1]
            )
        else:
            delta_t = 0.1
        # Update tracker and retrieve debug info
        st.session_state.tracks, track_debug = update_tracker(
            st.session_state.tracks,
            clusters,
            match_dist=track_match_dist,
            max_miss=max_miss,
            delta_t=delta_t,
            dyn_speed_thr=dyn_speed_thr,
        )

        # Build variable-resolution grid and memory savings
        grid_data = build_variable_grid(final_points, final_labels)
        uniform_cells_count, adaptive_cells_count, memory_saved_pct = memory_savings(
            grid_data, radius=100.0, finest_cell=0.05
        )

    pipeline_time = time.time() - t_start
    fps = 1.0 / pipeline_time if pipeline_time > 0 else 0.0
except Exception as ex:
    system_error = str(ex)

# ---------------------------------------------------------
# Primary Metrics Telemetry Row (extended to 7 columns)
# ---------------------------------------------------------
cols = st.columns(7)
(p1, p2, p3, p4, p5, p6, p7) = cols
with p1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">PIPELINE FPS</div>
      <div class="metric-value cyan">{fps:.1f}</div>
    </div>
    """, unsafe_allow_html=True)
with p2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">ADAPTIVE CELLS</div>
      <div class="metric-value">{adaptive_cells_count:,}</div>
    </div>
    """, unsafe_allow_html=True)
with p3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">UNIFORM CELLS (0.05m)</div>
      <div class="metric-value">{uniform_cells_count:,}</div>
    </div>
    """, unsafe_allow_html=True)
with p4:
    st.markdown(f"""
    <div class="metric-card highlight">
      <div class="metric-label">MEMORY SAVED</div>
      <div class="metric-value green">{memory_saved_pct:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)
# Tracking metric cards moved to later section – removed to avoid early reference to data_source
with p5:
    st.markdown(f""" 
    <div class="metric-card">
      <div class="metric-label">TRACKED OBJECTS</div>
      <div class="metric-value">{track_debug.get('active_tracks', 0) if data_source == 'KITTI Real LiDAR' else 0}</div>
    </div>
    """, unsafe_allow_html=True)
with p6:
    st.markdown(f""" 
    <div class="metric-card">
      <div class="metric-label">DYNAMIC OBJECTS</div>
      <div class="metric-value">{track_debug.get('dynamic', 0) if data_source == 'KITTI Real LiDAR' else 0}</div>
    </div>
    """, unsafe_allow_html=True)
with p7:
    st.markdown(f""" 
    <div class="metric-card">
      <div class="metric-label">STATIC OBJECTS</div>
      <div class="metric-value">{track_debug.get('static', 0) if data_source == 'KITTI Real LiDAR' else 0}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Plotly Track Visualisation (add after obstacle traces)
# ---------------------------------------------------------
if data_source == "KITTI Real LiDAR" and st.session_state.tracks:
    for trk in st.session_state.tracks:
        cx, cy = trk["centroid"][0], trk["centroid"][1]
        prev_cx, prev_cy = trk["previous_centroid"][0], trk["previous_centroid"][1]
        color = "#10b981" if trk["status"] == "static" else "#f59e0b"
        # ID annotation
        fig.add_trace(
            go.Scatter(
                x=[cx],
                y=[cy],
                mode="text",
                text=[f"ID {trk['track_id'] }"],
                textfont=dict(color=color, size=10, family="monospace"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Motion vector line (if track moved)
        fig.add_trace(
            go.Scatter(
                x=[prev_cx, cx],
                y=[prev_cy, cy],
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Optional short trail (last up to 5 points)
        if "trail" in trk and len(trk["trail"]) > 1:
            trail_x = [pt[0] for pt in trk["trail"]]
            trail_y = [pt[1] for pt in trk["trail"]]
            fig.add_trace(
                go.Scatter(
                    x=trail_x,
                    y=trail_y,
                    mode="lines",
                    line=dict(color="rgba(200,200,200,0.6)", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

# ---------------------------------------------------------
# Track Table expander
# ---------------------------------------------------------
if data_source == "KITTI Real LiDAR":
    with st.expander("TRACK TABLE", expanded=False):
        if st.session_state.tracks:
            df = pd.DataFrame([
                {
                    "Track ID": trk["track_id"],
                    "X": float(trk["centroid"][0]),
                    "Y": float(trk["centroid"][1]),
                    "Speed (m/s)": round(trk["speed"], 2),
                    "Status": trk["status"].capitalize(),
                }
                for trk in st.session_state.tracks
            ])
            st.dataframe(df)
        else:
            st.info("No active tracks.")

if "pedestrian_offset" not in st.session_state:
    st.session_state.pedestrian_offset = 1.2
if "synthetic_frame_id" not in st.session_state:
    st.session_state.synthetic_frame_id = 2
if "centroid_history" not in st.session_state:
    st.session_state.centroid_history = [(20.0, 5.0)]
if "kitti_frame_idx" not in st.session_state:
    st.session_state.kitti_frame_idx = 0


# ---------------------------------------------------------
# KITTI Dataset Discovery & Caching
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def discover_kitti_frames() -> List[str]:
    """Discover and sort all KITTI .bin files in the dataset directory."""
    candidates = [
        os.path.join("data", "kitti", "2011_09_26", "2011_09_26_drive_0005_sync", "velodyne_points", "data"),
        os.path.join("data", "kitti", "2011_09_26_drive_0005_sync", "velodyne_points", "data"),
        r"C:\Users\Yash Chhipa\OneDrive\Desktop\drdo\data\kitti\2011_09_26\2011_09_26_drive_0005_sync\velodyne_points\data",
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            files = sorted(glob.glob(os.path.join(candidate, "*.bin")))
            if files:
                return files
    return []


# ---------------------------------------------------------
# Cached Reference Frame Processing for Synthetic Mode
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_synthetic_reference(
    seed: int, distance_threshold: float, eps: float, min_points: int
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Optional[Tuple[float, float]]]:
    """Cache static reference frame 1 for synthetic mode."""
    pts1, _ = generate_frame(num_ground=8000, seed=seed, pedestrian_offset=0.0)
    g_mask1 = segment_ground(pts1, distance_threshold=distance_threshold)
    clusters1 = cluster_objects(pts1, ~g_mask1, eps=eps, min_points=min_points)

    dyn_c1 = None
    for c in clusters1:
        c_mean = c[:, :2].mean(axis=0)
        if np.hypot(c_mean[0] - 20.0, c_mean[1] - 5.0) < 3.0:
            dyn_c1 = (float(c_mean[0]), float(c_mean[1]))
            break

    return pts1, g_mask1, clusters1, dyn_c1


# ---------------------------------------------------------
# Sidebar: Perception Controls
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="font-family: ui-monospace, monospace; font-size: 0.95rem; font-weight: 800; color: #f8fafc; letter-spacing: 0.08em; padding: 4px 0 12px 0; border-bottom: 1px solid #1e293b; margin-bottom: 12px;">
      PERCEPTION CONTROL
    </div>
    """,
    unsafe_allow_html=True,
)

# 1. Data Source Selection
data_source = st.sidebar.radio(
    "Data Source",
    ["Synthetic", "KITTI Real LiDAR"],
    index=0,
    help="Select point cloud ingestion source: synthetic simulation or real Velodyne HDL-64E sequence.",
)

kitti_files = discover_kitti_frames() if data_source == "KITTI Real LiDAR" else []
# Initialise tracking‑related session state now that data_source exists
if "prev_data_source" not in st.session_state:
    st.session_state.prev_data_source = data_source
    # Load KITTI timestamps once
    if data_source == "KITTI Real LiDAR":
        if "kitti_timestamps" not in st.session_state:
            st.session_state.kitti_timestamps = load_kitti_timestamps()
        if not st.session_state.kitti_timestamps:
            st.caption("KITTI timestamps not found – using default 0.1 s interval.")

# 2. Simulation & Frame Controls
with st.sidebar.expander("SIMULATION & FRAMES", expanded=True):
    col_s1, col_s2 = st.columns(2)

    if data_source == "Synthetic":
        with col_s1:
            if st.button("Step Frame ⏭️", use_container_width=True):
                st.session_state.pedestrian_offset += 1.0
                st.session_state.synthetic_frame_id += 1
        with col_s2:
            if st.button("Reset 🔄", use_container_width=True):
                st.session_state.pedestrian_offset = 1.2
                st.session_state.synthetic_frame_id = 2
                st.session_state.centroid_history = [(20.0, 5.0)]

        pedestrian_offset = st.slider(
            "Pedestrian X Offset (m)",
            min_value=0.0,
            max_value=15.0,
            value=float(st.session_state.pedestrian_offset),
            step=0.2,
            help="Simulates dynamic pedestrian motion along the X axis.",
        )
        st.session_state.pedestrian_offset = pedestrian_offset

    else:  # KITTI Real LiDAR
        num_kitti = len(kitti_files)
        with col_s1:
            if st.button("Step Frame ⏭️", use_container_width=True):
                if num_kitti > 0:
                    st.session_state.kitti_frame_idx = (st.session_state.kitti_frame_idx + 1) % num_kitti
        with col_s2:
            if st.button("Reset 🔄", use_container_width=True):
                st.session_state.kitti_frame_idx = 0

        if num_kitti > 0:
            kitti_slider_val = st.slider(
                "KITTI Frame Index",
                min_value=0,
                max_value=num_kitti - 1,
                value=int(st.session_state.kitti_frame_idx),
                step=1,
            )
            st.session_state.kitti_frame_idx = kitti_slider_val
            current_kitti_filename = os.path.basename(kitti_files[st.session_state.kitti_frame_idx])
            st.caption(f"📁 `{current_kitti_filename}` ({st.session_state.kitti_frame_idx + 1} / {num_kitti})")
        else:
            st.warning("No KITTI .bin files located in `data/kitti/`.")

# (Removed duplicate perception parameters block - UI sliders now located earlier)

# 4. Layer Visibility Toggles
with st.sidebar.expander("VISUALIZATION", expanded=True):
    show_ground = st.checkbox("Ground Plane", value=True)
    show_obstacles = st.checkbox(
        "Static Obstacles" if data_source == "Synthetic" else "Detected Obstacles",
        value=True,
    )
    show_dynamic = st.checkbox("Dynamic Objects", value=True) if data_source == "Synthetic" else False
    show_rings = st.checkbox("Resolution Zones", value=True)
    show_trail = st.checkbox("Motion Trail", value=True) if data_source == "Synthetic" else False

# 5. Advanced / Debug Options
with st.sidebar.expander("ADVANCED / DEBUG", expanded=False):
    ground_downsample = st.checkbox(
        "Fast Ground Render (Visual Only)",
        value=True,
        help="Decoupled visual optimization. Downsamples ground points only in Plotly rendering layer for 60 FPS UI responsiveness. Does NOT affect full point cloud processing or grid calculations.",
    )
    st.caption("Point cloud pipeline processes 100% of data.")

# ---------------------------------------------------------
# Perception Pipeline Execution
# ---------------------------------------------------------
system_error: Optional[str] = None

# Pipeline output placeholders
total_raw_points = 0
ground_points_count = 0
non_ground_points_count = 0
num_obstacle_clusters = 0
num_dynamic_clusters = 0
adaptive_cells_count = 0
uniform_cells_count = 16000000
memory_saved_pct = 0.0
fps = 0.0
grid_data: Dict[Tuple[float, int, int], Dict[str, Any]] = {}
dynamic_centroid: Optional[Tuple[float, float]] = None

try:
    t_start = time.time()

    if data_source == "Synthetic":
        # 1. Fetch cached reference frame 1
        pts1, ground_mask1, clusters_frame1, dyn_ref_c = get_cached_synthetic_reference(
            seed=1,
            distance_threshold=ground_thresh,
            eps=dbscan_eps,
            min_points=dbscan_min_pts,
        )

        # 2. Ingest live frame 2
        pts_live, _ = generate_frame(
            num_ground=8000,
            seed=1,
            pedestrian_offset=st.session_state.pedestrian_offset,
        )

        # Range filter if specified
        if max_range > 0:
            r = np.hypot(pts_live[:, 0], pts_live[:, 1])
            pts_live = pts_live[r <= max_range]

        total_raw_points = len(pts_live)

        # 3. Ground segmentation
        ground_mask = segment_ground(pts_live, distance_threshold=ground_thresh)
        ground_points_count = int(np.sum(ground_mask))
        non_ground_points_count = total_raw_points - ground_points_count

        # 4. Object clustering
        clusters = cluster_objects(
            pts_live,
            ~ground_mask,
            eps=dbscan_eps,
            min_points=dbscan_min_pts,
        )

        # 5. Static vs Dynamic classification
        cluster_labels = classify_static_vs_dynamic(
            clusters_frame1,
            clusters,
            move_threshold=0.3,
        )

        # 6. Assemble labeled point cloud
        ground_pts = pts_live[ground_mask]
        ground_lbls = np.full(len(ground_pts), "ground", dtype=object)

        cluster_pts_list = []
        cluster_lbl_list = []
        for c, lbl in zip(clusters, cluster_labels):
            cluster_pts_list.append(c)
            cluster_lbl_list.append(np.full(len(c), lbl, dtype=object))
            if lbl == "dynamic_object":
                num_dynamic_clusters += 1
                c_mean = c[:, :2].mean(axis=0)
                dynamic_centroid = (float(c_mean[0]), float(c_mean[1]))
            elif lbl == "static_obstacle":
                num_obstacle_clusters += 1

        if cluster_pts_list:
            final_points = np.vstack([ground_pts] + cluster_pts_list)
            final_labels = np.concatenate([ground_lbls] + cluster_lbl_list)
        else:
            final_points = ground_pts
            final_labels = ground_lbls

        # Update dynamic motion history
        if dynamic_centroid is not None:
            curr_pt = dynamic_centroid
            if not st.session_state.centroid_history or np.hypot(
                curr_pt[0] - st.session_state.centroid_history[-1][0],
                curr_pt[1] - st.session_state.centroid_history[-1][1],
            ) > 0.05:
                st.session_state.centroid_history.append(curr_pt)
                if len(st.session_state.centroid_history) > 12:
                    st.session_state.centroid_history.pop(0)

        # 7. Adaptive Grid Construction
        grid_data = build_variable_grid(final_points, final_labels)

        # 8. Memory savings
        uniform_cells_count, adaptive_cells_count, memory_saved_pct = memory_savings(
            grid_data, radius=100.0, finest_cell=0.05
        )

    else:  # KITTI Real LiDAR
        if not kitti_files:
            raise RuntimeError("KITTI dataset files not found. Check path: data/kitti/...")

        kitti_file_path = kitti_files[st.session_state.kitti_frame_idx]

        # 1. Load real KITTI .bin frame
        pts_raw, _intensity = load_lidar_frame(kitti_file_path)

        # 2. Preprocessing: filter NaN / Infinite values
        valid_mask = np.isfinite(pts_raw).all(axis=1)
        pts_live = pts_raw[valid_mask]

        # Optional maximum-range filter (default 100m planar range)
        if max_range > 0:
            r = np.hypot(pts_live[:, 0], pts_live[:, 1])
            pts_live = pts_live[r <= max_range]

        total_raw_points = len(pts_live)

        # 3. Geometric Ground Segmentation (RANSAC)
        ground_mask = segment_ground(pts_live, distance_threshold=ground_thresh)
        ground_points_count = int(np.sum(ground_mask))
        non_ground_points_count = total_raw_points - ground_points_count

        # 4. Obstacle Clustering (DBSCAN on non-ground points)
        clusters = cluster_objects(
            pts_live,
            ~ground_mask,
            eps=dbscan_eps,
            min_points=dbscan_min_pts,
        )
        num_obstacle_clusters = len(clusters)

        # 5. Assemble labeled point cloud (Ground vs. Obstacle)
        # For single-frame raw LiDAR, no temporal motion is claimed yet
        ground_pts = pts_live[ground_mask]
        ground_lbls = np.full(len(ground_pts), "ground", dtype=object)

        cluster_pts_list = []
        cluster_lbl_list = []
        for c in clusters:
            cluster_pts_list.append(c)
            cluster_lbl_list.append(np.full(len(c), "static_obstacle", dtype=object))

        if cluster_pts_list:
            final_points = np.vstack([ground_pts] + cluster_pts_list)
            final_labels = np.concatenate([ground_lbls] + cluster_lbl_list)
        else:
            final_points = pts_live
            final_labels = np.full(len(final_points), "ground" if ground_points_count > 0 else "unknown", dtype=object)

        # 6. Build Variable-Resolution Grid (on 100% of live points)
        grid_data = build_variable_grid(final_points, final_labels)

        # 7. Compute live memory savings
        uniform_cells_count, adaptive_cells_count, memory_saved_pct = memory_savings(
            grid_data, radius=100.0, finest_cell=0.05
        )

    pipeline_time = time.time() - t_start
    fps = 1.0 / pipeline_time if pipeline_time > 0 else 0.0

except Exception as ex:
    system_error = str(ex)

# ---------------------------------------------------------
# Top Header & Status Indicator
# ---------------------------------------------------------
status_html = (
    """
    <div class="status-badge-online">
      <div class="status-dot-online"></div>
      SYSTEM ONLINE
    </div>
    """
    if system_error is None
    else f"""
    <div class="status-badge-error">
      ● SYSTEM ERROR
    </div>
    """
)

source_subtitle = (
    "Synthetic Simulation • 200m × 200m Domain"
    if data_source == "Synthetic"
    else f"KITTI Velodyne HDL-64E • {os.path.basename(kitti_files[st.session_state.kitti_frame_idx]) if kitti_files else 'Real LiDAR'}"
)

st.markdown(
    f"""
    <div class="telemetry-header">
      <div>
        <div class="telemetry-title">AUTONOMOUS PERCEPTION</div>
        <div class="telemetry-subtitle">Variable-Resolution LiDAR Grid</div>
        <div class="telemetry-tagline">{source_subtitle}</div>
      </div>
      <div>
        {status_html}
      </div>
    </div>
    <div class="pipeline-breadcrumb">
      LiDAR ({data_source}) → Ground Segmentation → Object Clustering → {'Tracking → ' if data_source == 'Synthetic' else ''}Adaptive Grid
    </div>
    """,
    unsafe_allow_html=True,
)

if system_error:
    st.error(f"Perception Pipeline Error: {system_error}")
    st.stop()

# ---------------------------------------------------------
# Primary Metrics Telemetry Row
# ---------------------------------------------------------
import pandas as pd
# After metric cards, add tracking metrics
p5, p6, p7 = st.columns(3)

with p5:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">TRACKED OBJECTS</div>
          <div class="metric-value magenta">{track_debug.get('active_tracks', 0):,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p6:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">DYNAMIC OBJECTS</div>
          <div class="metric-value orange">{track_debug.get('dynamic', 0):,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p7:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">STATIC OBJECTS</div>
          <div class="metric-value green">{track_debug.get('static', 0):,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Add caption for total KITTI frames (if KITTI mode)
if data_source == "KITTI Real LiDAR" and kitti_files:
    st.caption(f"KITTI sequence contains {len(kitti_files)} frames.")

# Track table expander (after plot)
if data_source == "KITTI Real LiDAR":
    with st.expander("TRACK TABLE", expanded=False):
        if st.session_state.tracks:
            df = pd.DataFrame([
                {
                    "Track ID": tr["track_id"],
                    "X": f"{tr['centroid'][0]:.2f}",
                    "Y": f"{tr['centroid'][1]:.2f}",
                    "Speed (m/s)": f"{tr['speed']:.2f}",
                    "Status": tr["status"].capitalize(),
                }
                for tr in st.session_state.tracks
            ])
            st.dataframe(df, hide_index=True)
        else:
            st.info("No active tracks.")


with p1:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">PIPELINE FPS</div>
          <div class="metric-value cyan">{fps:.1f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p2:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">ADAPTIVE CELLS</div>
          <div class="metric-value">{adaptive_cells_count:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p3:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">UNIFORM CELLS (0.05m)</div>
          <div class="metric-value">{uniform_cells_count:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p4:
    st.markdown(
        f"""
        <div class="metric-card highlight">
          <div class="metric-label">MEMORY SAVED</div>
          <div class="metric-value green">{memory_saved_pct:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Secondary Diagnostics Strip
if data_source == "Synthetic":
    diag_extra = f"""
      <div class="diag-item">STATIC CLUSTERS: <span class="val">{num_obstacle_clusters}</span></div>
      <div>•</div>
      <div class="diag-item">DYNAMIC CLUSTERS: <span class="val">{num_dynamic_clusters}</span></div>
    """
else:
    diag_extra = f"""
      <div class="diag-item">NON-GROUND POINTS: <span class="val">{non_ground_points_count:,}</span></div>
      <div>•</div>
      <div class="diag-item">DETECTED CLUSTERS: <span class="val">{num_obstacle_clusters}</span></div>
    """

st.markdown(
    f"""
    <div class="diag-strip">
      <div class="diag-item">TOTAL POINTS: <span class="val">{total_raw_points:,}</span></div>
      <div>•</div>
      <div class="diag-item">GROUND POINTS: <span class="val">{ground_points_count:,}</span></div>
      <div>•</div>
      {diag_extra}
      <div>•</div>
      <div class="diag-item">GRID DOMAIN: <span class="val">200m × 200m</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Main Visualization: LIVE LiDAR PERCEPTION
# ---------------------------------------------------------
vis_col_head, vis_col_badge = st.columns([3, 1])
with vis_col_head:
    st.markdown(
        f"""
        <div style="font-family: ui-monospace, monospace; font-size: 1.05rem; font-weight: 700; color: #f8fafc; letter-spacing: 0.04em;">
          LIVE LiDAR PERCEPTION — {'SYNTHETIC' if data_source == 'Synthetic' else 'KITTI REAL DATA'}
        </div>
        <div style="font-size: 0.74rem; color: #94a3b8;">
          Top-down spatial view • {'Synthetic Multi-Object Scene' if data_source == 'Synthetic' else 'Velodyne HDL-64E • 360° Scan'} • Adaptive resolution by range
        </div>
        """,
        unsafe_allow_html=True,
    )
with vis_col_badge:
    if data_source == "Synthetic":
        badge_text = f"FRAME <span style='color: #38bdf8; font-weight: 700;'>#{st.session_state.synthetic_frame_id:02d}</span> • OFFSET <span style='color: #f59e0b; font-weight: 700;'>+{st.session_state.pedestrian_offset:.1f}m</span>"
    else:
        num_kitti = len(kitti_files)
        fname = os.path.basename(kitti_files[st.session_state.kitti_frame_idx]) if kitti_files else ""
        badge_text = f"KITTI FRAME <span style='color: #38bdf8; font-weight: 700;'>{fname}</span> ({st.session_state.kitti_frame_idx + 1}/{num_kitti})"
    st.markdown(
        f"""
        <div style="text-align: right; font-family: ui-monospace, monospace; font-size: 0.75rem; color: #94a3b8; padding-top: 4px;">
          {badge_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

# Custom Legend Bar
if data_source == "Synthetic":
    legend_html = """
    <div class="legend-bar">
      <div class="legend-item"><span style="color: #10b981; font-size: 0.9rem;">●</span> Ground</div>
      <div class="legend-item"><span style="color: #ef4444; font-size: 0.9rem;">●</span> Static Obstacle</div>
      <div class="legend-item"><span style="color: #f59e0b; font-size: 0.9rem;">●</span> Dynamic Object</div>
      <div class="legend-item"><span style="color: #38bdf8; font-size: 0.9rem;">◆</span> LiDAR Sensor (0,0)</div>
      <div class="legend-item"><span style="color: #f59e0b; font-weight: 700;">╌╌</span> Motion Trail</div>
      <div class="legend-item"><span style="color: rgba(56, 189, 248, 0.7);">◯</span> Adaptive Zones (10m, 50m, 100m)</div>
    </div>
    """
else:
    legend_html = """
    <div class="legend-bar">
      <div class="legend-item"><span style="color: #10b981; font-size: 0.9rem;">●</span> Ground</div>
      <div class="legend-item"><span style="color: #ef4444; font-size: 0.9rem;">●</span> Detected Obstacles / Non-Ground</div>
      <div class="legend-item"><span style="color: #38bdf8; font-size: 0.9rem;">◆</span> LiDAR Sensor (0,0)</div>
      <div class="legend-item"><span style="color: rgba(56, 189, 248, 0.7);">◯</span> Adaptive Zones (10m, 50m, 100m)</div>
    </div>
    """

st.markdown(legend_html, unsafe_allow_html=True)

# Build Visualization Data Arrays
plot_ground_x: List[float] = []
plot_ground_y: List[float] = []
plot_ground_hover: List[str] = []

plot_obs_x: List[float] = []
plot_obs_y: List[float] = []
plot_obs_hover: List[str] = []

plot_dyn_x: List[float] = []
plot_dyn_y: List[float] = []
plot_dyn_hover: List[str] = []

# Decoupled visualization downsampling for ground points (preserves 60 FPS UI)
sample_step = 4 if data_source == "KITTI Real LiDAR" else 2
ground_counter = 0

for (cell_size, cell_x, cell_y), cdata in grid_data.items():
    label = cdata["label"]
    real_x = (cell_x + 0.5) * cell_size
    real_y = (cell_y + 0.5) * cell_size

    if label == "ground":
        if ground_downsample:
            ground_counter += 1
            if ground_counter % sample_step != 0:
                continue
        plot_ground_x.append(real_x)
        plot_ground_y.append(real_y)
        plot_ground_hover.append(
            f"<b>CLASS:</b> GROUND<br>"
            f"<b>POS:</b> ({real_x:.2f}m, {real_y:.2f}m)<br>"
            f"<b>HEIGHT:</b> {cdata['height']:.2f}m<br>"
            f"<b>CELL:</b> {cell_size * 100:.0f}cm ({cell_size:.2f}m)<br>"
            f"<b>POINTS:</b> {cdata['count']}"
        )
    elif label == "dynamic_object":
        plot_dyn_x.append(real_x)
        plot_dyn_y.append(real_y)
        plot_dyn_hover.append(
            f"<b>CLASS:</b> DYNAMIC OBJECT<br>"
            f"<b>POS:</b> ({real_x:.2f}m, {real_y:.2f}m)<br>"
            f"<b>HEIGHT:</b> {cdata['height']:.2f}m<br>"
            f"<b>CELL:</b> {cell_size * 100:.0f}cm<br>"
            f"<b>POINTS:</b> {cdata['count']}"
        )
    else:  # static_obstacle or non_ground
        plot_obs_x.append(real_x)
        plot_obs_y.append(real_y)
        plot_obs_hover.append(
            f"<b>CLASS:</b> {'STATIC OBSTACLE' if data_source == 'Synthetic' else 'DETECTED OBSTACLE'}<br>"
            f"<b>POS:</b> ({real_x:.2f}m, {real_y:.2f}m)<br>"
            f"<b>HEIGHT:</b> {cdata['height']:.2f}m<br>"
            f"<b>CELL:</b> {cell_size * 100:.0f}cm<br>"
            f"<b>POINTS:</b> {cdata['count']}"
        )

fig = go.Figure()

# 1. Ground points
if show_ground and plot_ground_x:
    fig.add_trace(
        go.Scatter(
            x=plot_ground_x,
            y=plot_ground_y,
            mode="markers",
            name="Ground",
            marker=dict(
                size=2.0 if data_source == "Synthetic" else 1.8,
                color="#10b981",
                opacity=0.35 if data_source == "Synthetic" else 0.28,
            ),
            text=plot_ground_hover,
            hoverinfo="text",
            showlegend=False,
        )
    )

# 2. Obstacles / Non-ground
if show_obstacles and plot_obs_x:
    fig.add_trace(
        go.Scatter(
            x=plot_obs_x,
            y=plot_obs_y,
            mode="markers",
            name="Obstacles",
            marker=dict(
                size=7.5 if data_source == "Synthetic" else 4.0,
                color="#ef4444",
                opacity=0.95,
                line=dict(width=1.0 if data_source == "Synthetic" else 0.5, color="#ffffff"),
            ),
            text=plot_obs_hover,
            hoverinfo="text",
            showlegend=False,
        )
    )

# 3. Dynamic Objects (Synthetic mode only)
if data_source == "Synthetic" and show_dynamic and plot_dyn_x:
    fig.add_trace(
        go.Scatter(
            x=plot_dyn_x,
            y=plot_dyn_y,
            mode="markers",
            name="Dynamic Object",
            marker=dict(
                size=11.5,
                color="#f59e0b",
                opacity=1.0,
                line=dict(width=2.0, color="#ffffff"),
            ),
            text=plot_dyn_hover,
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Dynamic annotation
    if dynamic_centroid is not None:
        fig.add_annotation(
            x=dynamic_centroid[0],
            y=dynamic_centroid[1],
            text="<b>DYNAMIC</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor="#f59e0b",
            ax=35,
            ay=-25,
            font=dict(color="#f59e0b", size=10, family="monospace"),
            bgcolor="rgba(15, 23, 42, 0.9)",
            bordercolor="#f59e0b",
            borderwidth=1,
        )

# 4. Motion Trail (Synthetic mode only)
if data_source == "Synthetic" and show_trail and len(st.session_state.centroid_history) > 1 and show_dynamic:
    t_x = [pt[0] for pt in st.session_state.centroid_history]
    t_y = [pt[1] for pt in st.session_state.centroid_history]
    fig.add_trace(
        go.Scatter(
            x=t_x,
            y=t_y,
            mode="lines+markers",
            name="Motion Trail",
            line=dict(color="#f59e0b", width=2.0, dash="dash"),
            marker=dict(size=4.5, color="#fbbf24"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

# 5. Sensor Origin Marker & Forward Heading
fig.add_trace(
    go.Scatter(
        x=[0.0],
        y=[0.0],
        mode="markers",
        name="Sensor (0,0)",
        marker=dict(size=13, color="#38bdf8", symbol="diamond", line=dict(width=1.5, color="#ffffff")),
        text=["<b>LiDAR SENSOR (ORIGIN)</b><br>Coordinates: (0.00, 0.00)<br>Sensor: Velodyne HDL-64E" if data_source == "KITTI Real LiDAR" else "<b>LiDAR SENSOR (ORIGIN)</b><br>Coordinates: (0.00, 0.00)"],
        hoverinfo="text",
        showlegend=False,
    )
)

# Forward heading indicator ray (+X direction)
fig.add_trace(
    go.Scatter(
        x=[0.0, 14.0],
        y=[0.0, 0.0],
        mode="lines",
        name="Heading",
        line=dict(color="rgba(56, 189, 248, 0.5)", width=1.5, dash="dot"),
        hoverinfo="skip",
        showlegend=False,
    )
)

fig.add_annotation(
    x=0,
    y=0,
    text="<b>LiDAR SENSOR</b>",
    showarrow=True,
    arrowhead=1,
    arrowsize=0.8,
    arrowcolor="#38bdf8",
    ax=-45,
    ay=-25,
    font=dict(color="#38bdf8", size=10, family="monospace"),
    bgcolor="rgba(15, 23, 42, 0.9)",
    bordercolor="#38bdf8",
    borderwidth=1,
)

# 6. Adaptive Resolution Range Rings
if show_rings:
    # 10m Ring (Tier 1: 5 cm)
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=-10,
        y0=-10,
        x1=10,
        y1=10,
        line=dict(color="rgba(56, 189, 248, 0.45)", width=1.2, dash="dash"),
    )
    # 50m Ring (Tier 2: 20 cm)
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=-50,
        y0=-50,
        x1=50,
        y1=50,
        line=dict(color="rgba(56, 189, 248, 0.35)", width=1.2, dash="dash"),
    )
    # 100m Ring (Tier 3: 50 cm)
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=-100,
        y0=-100,
        x1=100,
        y1=100,
        line=dict(color="rgba(56, 189, 248, 0.25)", width=1.2, dash="dash"),
    )

    # Edge callout annotations (+X edge)
    fig.add_annotation(
        x=10.0,
        y=0.0,
        text="<b>5 CM</b><br><span style='font-size:8px'>0–10m</span>",
        showarrow=False,
        xanchor="left",
        xshift=4,
        font=dict(color="#38bdf8", size=9, family="monospace"),
        bgcolor="rgba(15, 23, 42, 0.85)",
        bordercolor="rgba(56, 189, 248, 0.5)",
        borderwidth=1,
    )
    fig.add_annotation(
        x=50.0,
        y=0.0,
        text="<b>20 CM</b><br><span style='font-size:8px'>10–50m</span>",
        showarrow=False,
        xanchor="left",
        xshift=4,
        font=dict(color="#38bdf8", size=9, family="monospace"),
        bgcolor="rgba(15, 23, 42, 0.85)",
        bordercolor="rgba(56, 189, 248, 0.4)",
        borderwidth=1,
    )
    fig.add_annotation(
        x=100.0,
        y=0.0,
        text="<b>50 CM</b><br><span style='font-size:8px'>50–100m</span>",
        showarrow=False,
        xanchor="left",
        xshift=4,
        font=dict(color="#38bdf8", size=9, family="monospace"),
        bgcolor="rgba(15, 23, 42, 0.85)",
        bordercolor="rgba(56, 189, 248, 0.3)",
        borderwidth=1,
    )

fig.update_layout(
    xaxis=dict(
        title="X (m)",
        range=[-105, 105],
        zeroline=True,
        zerolinecolor="#1e293b",
        gridcolor="#0f172a",
        title_font=dict(family="monospace", size=11, color="#64748b"),
        tickfont=dict(family="monospace", size=9, color="#64748b"),
    ),
    yaxis=dict(
        title="Y (m)",
        range=[-105, 105],
        scaleanchor="x",
        scaleratio=1,
        zeroline=True,
        zerolinecolor="#1e293b",
        gridcolor="#0f172a",
        title_font=dict(family="monospace", size=11, color="#64748b"),
        tickfont=dict(family="monospace", size=9, color="#64748b"),
    ),
    height=690,
    plot_bgcolor="#060912",
    paper_bgcolor="#060912",
    margin=dict(l=20, r=20, t=20, b=20),
    hoverlabel=dict(
        bgcolor="#0c1222",
        bordercolor="#38bdf8",
        font=dict(family="monospace", size=10, color="#f8fafc"),
    ),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------
# Information Panel: ADAPTIVE GRID EFFICIENCY
# ---------------------------------------------------------
st.markdown(
    """
    <div style="font-family: ui-monospace, monospace; font-size: 0.95rem; font-weight: 700; color: #f8fafc; letter-spacing: 0.04em; margin-top: 14px; margin-bottom: 8px;">
      ADAPTIVE GRID EFFICIENCY
    </div>
    """,
    unsafe_allow_html=True,
)

eff_col1, eff_col2 = st.columns([1, 1])

with eff_col1:
    st.markdown(
        f"""
        <div style="background: #080d1a; border: 1px solid #162032; border-radius: 6px; padding: 14px 16px; font-family: ui-monospace, monospace; font-size: 0.8rem; line-height: 1.6;">
          <div style="color: #94a3b8; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">Spatial Memory Reduction Summary</div>
          <div>• <b>Uniform 5 cm Grid:</b> <span style="color: #ef4444;">16,000,000 cells</span> ((200m / 0.05m)²)</div>
          <div>• <b>Adaptive Grid:</b> <span style="color: #38bdf8;">{adaptive_cells_count:,} cells</span> (Multi-tier allocation)</div>
          <div>• <b>Cell Reduction:</b> <span style="color: #10b981; font-weight: 700;">{memory_saved_pct:.4f}%</span></div>
          <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #1e293b; color: #64748b; font-size: 0.74rem;">
            Tier 1 (0–10m): <b>5cm</b> &nbsp;|&nbsp; Tier 2 (10–50m): <b>20cm</b> &nbsp;|&nbsp; Tier 3 (50–100m): <b>50cm</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with eff_col2:
    pct_used = (adaptive_cells_count / uniform_cells_count) * 100.0 if uniform_cells_count > 0 else 0.0
    st.markdown(
        f"""
        <div style="background: #080d1a; border: 1px solid #162032; border-radius: 6px; padding: 14px 16px; font-family: ui-monospace, monospace; font-size: 0.78rem;">
          <div style="color: #94a3b8; font-weight: 700; text-transform: uppercase; margin-bottom: 8px;">Memory Footprint Comparison</div>
          
          <div style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
              <span style="color: #e2e8f0;">Uniform 5cm Grid</span>
              <span style="color: #ef4444;">16,000,000 cells (100.0%)</span>
            </div>
            <div style="background: #1e293b; border-radius: 3px; height: 12px; width: 100%;">
              <div style="background: #ef4444; width: 100%; height: 12px; border-radius: 3px;"></div>
            </div>
          </div>

          <div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
              <span style="color: #e2e8f0;">Adaptive Variable Grid</span>
              <span style="color: #10b981;">{adaptive_cells_count:,} cells ({pct_used:.3f}%)</span>
            </div>
            <div style="background: #1e293b; border-radius: 3px; height: 12px; width: 100%;">
              <div style="background: #10b981; width: {max(0.8, pct_used):.2f}%; height: 12px; border-radius: 3px;"></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Pipeline Architecture (Collapsible)
# ---------------------------------------------------------
with st.expander("PIPELINE", expanded=False):
    if data_source == "Synthetic":
        flowchart = """
        Synthetic LiDAR Point Cloud (data_gen.generate_frame)
                        │
                        ▼
        Ground Segmentation (RANSAC Plane Fitting • terrain_segmentation.py)
                        │
                        ▼
        Object Clustering (Open3D DBSCAN • object_clustering.py)
                        │
                        ▼
        Static / Dynamic Classification (Inter-frame Centroid Displacement)
                        │
                        ▼
        Tracking (Dynamic Object Motion Trail & Heading Vector)
                        │
                        ▼
        Variable-Resolution Grid (Multi-Tier Cell Allocation • variable_resolution_grid.py)
        """
    else:
        flowchart = """
        KITTI Real Velodyne .bin (kitti_loader.load_lidar_frame)
                        │
                        ▼
        Point Cloud Preprocessing (NaN / Infinite Value Removal & Range Filter)
                        │
                        ▼
        Ground Segmentation (RANSAC Plane Fitting • terrain_segmentation.py)
                        │
                        ▼
        Obstacle Clustering (Open3D DBSCAN • object_clustering.py)
                        │
                        ▼
        Variable-Resolution Grid (Multi-Tier Cell Allocation • variable_resolution_grid.py)
        """

    st.markdown(f"```\n{flowchart}\n```")

# ---------------------------------------------------------
# System Status & Technical Context Note
# ---------------------------------------------------------
st.markdown(
    f"""
    <div style="margin-top: 14px; padding: 10px 14px; background: #080c16; border: 1px solid #162032; border-radius: 4px; font-family: ui-monospace, monospace; font-size: 0.72rem; color: #64748b;">
      <b style="color: #94a3b8;">SYSTEM NOTE:</b> Currently operating in <b>{data_source.upper()}</b> mode. The variable-resolution spatial grid allocates fine resolution (5cm) for near-field proximity (<10m), transitioning to 20cm and 50cm at range, producing >99.7% memory reductions across both synthetic and real Velodyne point clouds.
    </div>
    """,
    unsafe_allow_html=True,
)
