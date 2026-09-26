from kitti_loader import load_lidar_frame
from variable_resolution_grid import build_variable_grid, memory_savings


file_path = r"C:\Users\Yash Chhipa\OneDrive\Desktop\drdo\data\kitti\2011_09_26\2011_09_26_drive_0005_sync\velodyne_points\data\0000000000.bin"

points, intensity = load_lidar_frame(file_path)

print("Loaded LiDAR points:", len(points))

grid = build_variable_grid(points)

uniform_cnt, adaptive_cnt, savings_pct = memory_savings(
    grid,
    radius=100.0,
    finest_cell=0.05,
)

print("Adaptive grid cells:", adaptive_cnt)
print("Uniform grid cells:", uniform_cnt)
print("Memory reduction:", f"{savings_pct:.2f}%")