"""Variable-resolution multi-tier grid module for 3D point cloud mapping.

Provides:
- TIERS: (max_distance_metres, cell_size_metres) configuration.
- cell_size_for_distance(): Selects appropriate grid cell size by distance.
- build_variable_grid(): Constructs an adaptive occupancy/elevation grid.
- memory_savings(): Computes storage reduction compared to uniform finest grid.
"""

from typing import Dict, Tuple, Any
import numpy as np

# List of (max_distance_metres, cell_size_metres) tuples
TIERS = [
    (10, 0.05),
    (50, 0.20),
    (100, 0.50),
]


def cell_size_for_distance(r: float) -> float:
    """Return the cell size for the first tier whose max_distance is >= r.

    Defaults to the coarsest size beyond 100m.

    Args:
        r: Planar distance from the origin in metres.

    Returns:
        cell_size: Grid cell size in metres.
    """
    for max_dist, cell_size in TIERS:
        if max_dist >= r:
            return cell_size
    return TIERS[-1][1]


def build_variable_grid(
    points: np.ndarray,
    labels: np.ndarray | None = None,
) -> Dict[Tuple[float, int, int], Dict[str, Any]]:
    """Build a variable-resolution grid from 3D point cloud data.

    Labels are optional. If no labels are provided, points are treated
    as unlabeled LiDAR measurements.
    """

    grid: Dict[Tuple[float, int, int], Dict[str, Any]] = {}

    if labels is None:
        labels = np.full(len(points), "unknown", dtype=object)

    for (x, y, z), label in zip(points, labels):
        r = float(np.hypot(x, y))

        cell_size = cell_size_for_distance(r)

        cell_x = int(np.floor(x / cell_size))
        cell_y = int(np.floor(y / cell_size))

        key = (cell_size, cell_x, cell_y)

        z_val = float(z)

        if key not in grid:
            grid[key] = {
                "height": z_val,
                "label": str(label),
                "count": 1,
            }

        else:
            cell = grid[key]

            cell["count"] += 1

            if z_val > cell["height"]:
                cell["height"] = z_val
                cell["label"] = str(label)

    return grid

def memory_savings(
    grid: Dict[Tuple[float, int, int], Any],
    radius: float = 100.0,
    finest_cell: float = 0.05,
) -> Tuple[int, int, float]:
    """Compute memory reduction comparing adaptive grid cells used vs uniform finest grid.

    Args:
        grid: Adaptive grid dictionary from build_variable_grid().
        radius: Half-width/extent of the square bounding area in metres (covers 2*radius x 2*radius).
        finest_cell: Cell size of the finest uniform grid in metres.

    Returns:
        (uniform_cell_count, adaptive_cell_count, reduction_percentage)
    """
    grid_side_cells = int(np.ceil((2.0 * radius) / finest_cell))
    uniform_cell_count = grid_side_cells * grid_side_cells
    adaptive_cell_count = len(grid)

    reduction_percentage = (
        (uniform_cell_count - adaptive_cell_count) / uniform_cell_count
    ) * 100.0

    return uniform_cell_count, adaptive_cell_count, reduction_percentage


if __name__ == "__main__":
    from data_gen import generate_frame

    pts, lbls = generate_frame(num_ground=8000, seed=42)
    grid = build_variable_grid(pts, lbls)

    uniform_cnt, adapt_cnt, savings_pct = memory_savings(
        grid, radius=100.0, finest_cell=0.05
    )

    print(f"Total points mapped: {len(pts)}")
    print(f"Adaptive grid cells allocated: {adapt_cnt:,}")
    print(f"Equivalent uniform finest grid (0.05m) cells: {uniform_cnt:,}")
    print(f"Memory reduction: {savings_pct:.4f}%")

    # Sample cell breakdown by tier
    tier_counts = {tier[1]: 0 for tier in TIERS}
    for cell_size, _, _ in grid.keys():
        tier_counts[cell_size] += 1
    print("\nOccupied cells per resolution tier:")
    for cell_size, count in tier_counts.items():
        print(f"  Cell size {cell_size:0.2f}m: {count:,} cells")
