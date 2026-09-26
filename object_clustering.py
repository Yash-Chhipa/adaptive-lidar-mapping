"""Object clustering and tracking module using Open3D DBSCAN and centroid matching.

Provides:
- cluster_objects(): Groups non-ground 3D points into clusters using DBSCAN.
- classify_static_vs_dynamic(): Classifies frame 2 clusters as static vs. dynamic
  based on x,y centroid displacement relative to frame 1.
"""

from typing import List
import numpy as np
import open3d as o3d


def cluster_objects(
    points: np.ndarray,
    non_ground_mask: np.ndarray,
    eps: float = 0.6,
    min_points: int = 15,
) -> List[np.ndarray]:
    """Group non-ground points into distinct clusters using DBSCAN.

    Args:
        points: numpy array of shape (N, 3), 3D point coordinates.
        non_ground_mask: Boolean numpy array of shape (N,) indicating non-ground points.
        eps: DBSCAN cluster neighborhood distance threshold.
        min_points: Minimum number of points required to form a dense region.

    Returns:
        clusters: List of numpy arrays, each of shape (M_i, 3), containing
                  points belonging to each discovered cluster (noise points labeled -1
                  are excluded).
    """
    pts = points[non_ground_mask]
    if len(pts) == 0:
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )

    valid_labels = [lbl for lbl in np.unique(labels) if lbl >= 0]
    clusters = [pts[labels == lbl] for lbl in valid_labels]
    return clusters


def classify_static_vs_dynamic(
    clusters_frame1: List[np.ndarray],
    clusters_frame2: List[np.ndarray],
    move_threshold: float = 0.3,
) -> List[str]:
    """Classify clusters in frame 2 as static or dynamic based on centroid distance to frame 1.

    Computes the (x, y) centroid of every cluster in both frames, finds each frame 2
    cluster's nearest centroid in frame 1 by Euclidean distance, and assigns
    'dynamic_object' if min distance > move_threshold, else 'static_obstacle'.

    Args:
        clusters_frame1: List of cluster point arrays from frame 1.
        clusters_frame2: List of cluster point arrays from frame 2.
        move_threshold: Minimum 2D Euclidean distance shift to consider a cluster dynamic.

    Returns:
        labels: List of string labels ('static_obstacle' or 'dynamic_object')
                in the same order as clusters_frame2.
    """
    if len(clusters_frame2) == 0:
        return []

    # If frame 1 has no clusters, all frame 2 clusters are considered dynamic/new
    if len(clusters_frame1) == 0:
        return ["dynamic_object"] * len(clusters_frame2)

    # Compute (x, y) centroids
    centroids1 = np.array([c[:, :2].mean(axis=0) for c in clusters_frame1])  # (K1, 2)
    centroids2 = np.array([c[:, :2].mean(axis=0) for c in clusters_frame2])  # (K2, 2)

    labels = []
    for c2 in centroids2:
        dists = np.linalg.norm(centroids1 - c2, axis=1)
        min_dist = float(np.min(dists))
        if min_dist > move_threshold:
            labels.append("dynamic_object")
        else:
            labels.append("static_obstacle")

    return labels


if __name__ == "__main__":
    from data_gen import generate_frame
    from terrain_segmentation import segment_ground

    # Frame 1: dynamic object at offset 0.0
    pts1, true_lbl1 = generate_frame(num_ground=8000, seed=42, pedestrian_offset=0.0)
    ground_mask1 = segment_ground(pts1, distance_threshold=0.08)
    clusters1 = cluster_objects(pts1, ~ground_mask1, eps=0.6, min_points=15)

    # Frame 2: dynamic object moved by 1.2m
    pts2, true_lbl2 = generate_frame(num_ground=8000, seed=43, pedestrian_offset=1.2)
    ground_mask2 = segment_ground(pts2, distance_threshold=0.08)
    clusters2 = cluster_objects(pts2, ~ground_mask2, eps=0.6, min_points=15)

    classified_labels = classify_static_vs_dynamic(
        clusters1, clusters2, move_threshold=0.3
    )

    print(f"Found {len(clusters1)} clusters in Frame 1")
    print(f"Found {len(clusters2)} clusters in Frame 2")
    for i, (cluster, label) in enumerate(zip(clusters2, classified_labels)):
        centroid = cluster[:, :2].mean(axis=0)
        print(
            f"Cluster {i}: size={len(cluster)}, centroid_xy=({centroid[0]:.2f}, {centroid[1]:.2f}) -> {label}"
        )
