import os, glob, time, numpy as np
from kitti_loader import load_lidar_frame
from terrain_segmentation import segment_ground
from object_clustering import cluster_objects
from variable_resolution_grid import build_variable_grid, memory_savings

# Locate KITTI data directory (adjust if needed)
kitti_dir = r"c:/Users/Yash Chhipa/OneDrive/Desktop/drdo/data/kitti/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data"
files = sorted(glob.glob(os.path.join(kitti_dir, "*.bin")))
if not files:
    raise FileNotFoundError(f"No .bin files found in {kitti_dir}")

first_file = files[0]
print(f"Loading first KITTI frame: {first_file}")

# Load frame
load_start = time.time()
points, intensity = load_lidar_frame(first_file)
load_time = (time.time() - load_start) * 1000
print(f"Loaded {points.shape[0]} points in {load_time:.2f} ms")

# Ground segmentation
seg_start = time.time()
ground_mask = segment_ground(points, distance_threshold=0.15)
seg_time = (time.time() - seg_start) * 1000
print(f"Ground segmentation: {np.sum(ground_mask)} ground points, {points.shape[0] - np.sum(ground_mask)} non-ground, time {seg_time:.2f} ms")

# Clustering on non-ground points
cluster_start = time.time()
clusters = cluster_objects(points, ~ground_mask, eps=0.8, min_points=25)
cluster_time = (time.time() - cluster_start) * 1000
non_ground_points = points[~ground_mask]
print(f"Clustering produced {len(clusters)} clusters from {non_ground_points.shape[0]} points in {cluster_time:.2f} ms")
# Compute noise points (points not assigned to any cluster)
assigned = sum(len(c) for c in clusters)
noise = non_ground_points.shape[0] - assigned
print(f"Noise points (ignored): {noise}")
if clusters:
    sizes = [c.shape[0] for c in clusters]
    print(f"Largest cluster size: {max(sizes)}, smallest cluster size: {min(sizes)}")

# Build variable-resolution grid for all points
grid_start = time.time()
grid = build_variable_grid(points, labels=None)
grid_time = (time.time() - grid_start) * 1000
uniform_cells, adaptive_cells, savings = memory_savings(grid, radius=100.0, finest_cell=0.05)
print(f"Variable grid built in {grid_time:.2f} ms. Adaptive cells: {adaptive_cells}, Uniform cells (0.05m): {uniform_cells}, Savings: {savings:.2f}%")

print("--- Summary ---")
print(f"Total pipeline time (excluding import): {load_time + seg_time + cluster_time + grid_time:.2f} ms")
