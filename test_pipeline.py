"""Test script to verify end-to-end ground segmentation, clustering, and dynamic object tracking.

Generates two frames with different pedestrian offsets, clusters the non-ground points,
and verifies that exactly one 'dynamic_object' is detected while all other clusters
are classified as 'static_obstacle'.
"""

from collections import Counter
from data_gen import generate_frame
from terrain_segmentation import segment_ground
from object_clustering import cluster_objects, classify_static_vs_dynamic


def main():
    print("=== Generating Frames ===")
    # Generate Frame 1 (pedestrian at baseline x = 20.0)
    pts1, _ = generate_frame(num_ground=8000, seed=42, pedestrian_offset=0.0)
    # Generate Frame 2 (pedestrian displaced by 1.5m along x-axis to x = 21.5)
    pts2, _ = generate_frame(num_ground=8000, seed=43, pedestrian_offset=1.5)

    print("=== Segmenting Ground Plane ===")
    ground_mask1 = segment_ground(pts1, distance_threshold=0.08)
    ground_mask2 = segment_ground(pts2, distance_threshold=0.08)

    print("=== Clustering Non-Ground Objects (DBSCAN) ===")
    clusters1 = cluster_objects(pts1, ~ground_mask1, eps=0.6, min_points=15)
    clusters2 = cluster_objects(pts2, ~ground_mask2, eps=0.6, min_points=15)

    print(f"Frame 1: Discovered {len(clusters1)} clusters")
    print(f"Frame 2: Discovered {len(clusters2)} clusters")

    print("\n=== Classifying Static vs Dynamic Obstacles in Frame 2 ===")
    labels = classify_static_vs_dynamic(clusters1, clusters2, move_threshold=0.3)

    for i, (cluster, label) in enumerate(zip(clusters2, labels)):
        cx, cy = cluster[:, :2].mean(axis=0)
        print(f"Cluster {i}: centroid=({cx:6.2f}, {cy:6.2f}), points={len(cluster):3d} -> label: '{label}'")

    counts = Counter(labels)
    print("\n=== Classification Summary ===")
    print(f"Static obstacles: {counts['static_obstacle']}")
    print(f"Dynamic objects:  {counts['dynamic_object']}")

    # Validation
    assert counts["dynamic_object"] == 1, f"Expected 1 dynamic_object, got {counts['dynamic_object']}"
    assert counts["static_obstacle"] == len(clusters2) - 1, "Expected all remaining clusters to be static_obstacle"
    print("\nVerification Passed: Exactly one 'dynamic_object' and all others 'static_obstacle'.")


if __name__ == "__main__":
    main()
