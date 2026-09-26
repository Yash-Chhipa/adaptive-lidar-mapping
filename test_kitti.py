import numpy as np

file_path = r"C:\Users\Yash Chhipa\OneDrive\Desktop\drdo\data\kitti\2011_09_26\2011_09_26_drive_0005_sync\velodyne_points\data\0000000000.bin"

points = np.fromfile(file_path, dtype=np.float32)

points = points.reshape(-1, 4)

print("Shape:", points.shape)
print("First 5 points:")
print(points[:5])

print("\nX range:", points[:, 0].min(), points[:, 0].max())
print("Y range:", points[:, 1].min(), points[:, 1].max())
print("Z range:", points[:, 2].min(), points[:, 2].max())
print("Intensity range:", points[:, 3].min(), points[:, 3].max())