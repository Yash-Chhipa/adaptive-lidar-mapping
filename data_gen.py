"""Data generation module for synthetic 3D point cloud scenes.

Provides generate_frame() for producing synthetic LiDAR/point cloud frames
with ground plane, static obstacles, and dynamic pedestrian objects.
"""

from typing import Tuple
import numpy as np


def generate_frame(
    num_ground: int = 8000,
    seed: int = 0,
    pedestrian_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic 3D point cloud frame and corresponding point labels.

    Args:
        num_ground: Number of points on the ground plane.
        seed: Random seed for reproducibility using np.random.Generator.
        pedestrian_offset: Offset along the x-axis for the dynamic object.

    Returns:
        points: numpy array of shape (N, 3), float32 point coordinates (x, y, z).
        labels: numpy array of shape (N,), string labels for each point
                ('ground', 'static_obstacle', 'dynamic_object').
    """
    rng = np.random.default_rng(seed)

    # 1. Ground plane: 200m x 200m centered at origin (x in [-100, 100], y in [-100, 100])
    ground_x = rng.uniform(-100.0, 100.0, size=(num_ground, 1)).astype(np.float32)
    ground_y = rng.uniform(-100.0, 100.0, size=(num_ground, 1)).astype(np.float32)
    ground_z = rng.normal(loc=0.0, scale=0.03, size=(num_ground, 1)).astype(np.float32)
    ground_pts = np.hstack([ground_x, ground_y, ground_z])
    ground_labels = np.full(num_ground, "ground", dtype=object)

    # Helper for generating points inside a box obstacle
    def _sample_box(center, size, count):
        cx, cy, cz = center
        dx, dy, dz = size
        low = np.array([cx - dx / 2.0, cy - dy / 2.0, cz - dz / 2.0], dtype=np.float32)
        high = np.array([cx + dx / 2.0, cy + dy / 2.0, cz + dz / 2.0], dtype=np.float32)
        return rng.uniform(low, high, size=(count, 3)).astype(np.float32)

    # 2. Three static box-shaped obstacles at increasing distances from the origin:
    # - Obstacle 1: distance ~ 7.8m (within 5-10m)
    obs1_pts = _sample_box(center=(6.0, 5.0, 0.75), size=(2.0, 2.0, 1.5), count=400)
    # - Obstacle 2: distance = 40m (32^2 + 24^2 = 1600 = 40^2)
    obs2_pts = _sample_box(center=(32.0, 24.0, 1.0), size=(2.5, 2.5, 2.0), count=400)
    # - Obstacle 3: distance = 90m (54^2 + 72^2 = 8100 = 90^2, within 85-95m)
    obs3_pts = _sample_box(center=(-54.0, 72.0, 1.0), size=(2.5, 2.5, 2.0), count=400)

    static_pts = np.vstack([obs1_pts, obs2_pts, obs3_pts])
    static_labels = np.full(len(static_pts), "static_obstacle", dtype=object)

    # 3. Dynamic object: small box around (20 + pedestrian_offset, 5)
    dyn_center = (20.0 + float(pedestrian_offset), 5.0, 0.9)
    dyn_size = (0.8, 0.8, 1.8)
    dyn_pts = _sample_box(center=dyn_center, size=dyn_size, count=200)
    dyn_labels = np.full(len(dyn_pts), "dynamic_object", dtype=object)

    # Combine all components
    points = np.vstack([ground_pts, static_pts, dyn_pts]).astype(np.float32)
    labels = np.concatenate([ground_labels, static_labels, dyn_labels]).astype(str)

    return points, labels


if __name__ == "__main__":
    pts1, lbl1 = generate_frame(num_ground=8000, seed=42, pedestrian_offset=0.0)
    pts2, lbl2 = generate_frame(num_ground=8000, seed=42, pedestrian_offset=2.5)

    print(f"Frame 1 points shape: {pts1.shape}, dtype: {pts1.dtype}")
    print(f"Frame 1 labels shape: {lbl1.shape}, dtype: {lbl1.dtype}")
    unique_labels, counts = np.unique(lbl1, return_counts=True)
    for lbl, count in zip(unique_labels, counts):
        print(f"  {lbl}: {count} points")

    dyn_mask1 = lbl1 == "dynamic_object"
    dyn_mask2 = lbl2 == "dynamic_object"
    mean_dyn1 = pts1[dyn_mask1].mean(axis=0)
    mean_dyn2 = pts2[dyn_mask2].mean(axis=0)
    print(f"Dynamic object centroid (offset 0.0): {mean_dyn1}")
    print(f"Dynamic object centroid (offset 2.5): {mean_dyn2}")
    print(f"X shift: {mean_dyn2[0] - mean_dyn1[0]:.4f}m (expected ~2.5m)")
