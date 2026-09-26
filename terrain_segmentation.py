"""Terrain segmentation module using Open3D RANSAC plane fitting.

Provides segment_ground() to separate ground plane points from obstacles
in 3D point cloud data.
"""

import numpy as np
import open3d as o3d


def segment_ground(points: np.ndarray, distance_threshold: float = 0.08) -> np.ndarray:
    """RANSAC-fit a ground plane to 3D points and return an inlier mask.

    Args:
        points: numpy array of shape (N, 3), 3D point coordinates.
        distance_threshold: Maximum distance from the fitted plane for a point
            to be considered an inlier. Default is 0.08m.

    Returns:
        mask: Boolean numpy array of shape (N,) where True indicates
            the point belongs to the ground plane, and False otherwise.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    _plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=200,
    )

    mask = np.zeros(len(points), dtype=bool)
    mask[inliers] = True
    return mask


if __name__ == "__main__":
    from data_gen import generate_frame

    pts, labels = generate_frame(num_ground=8000, seed=42)
    ground_mask = segment_ground(pts, distance_threshold=0.08)

    true_ground = labels == "ground"
    accuracy = np.mean(ground_mask == true_ground)
    print(f"Total points: {len(pts)}")
    print(f"Detected ground points: {np.sum(ground_mask)}")
    print(f"Actual ground points: {np.sum(true_ground)}")
    print(f"Ground segmentation accuracy: {accuracy * 100:.2f}%")
