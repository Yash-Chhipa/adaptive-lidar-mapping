from kitti_loader import load_lidar_frame


file_path = r"C:\Users\Yash Chhipa\OneDrive\Desktop\drdo\data\kitti\2011_09_26\2011_09_26_drive_0005_sync\velodyne_points\data\0000000000.bin"

points, intensity = load_lidar_frame(file_path)

print("Point cloud shape:", points.shape)
print("Intensity shape:", intensity.shape)

print("\nFirst 5 XYZ points:")
print(points[:5])

print("\nFirst 5 intensities:")
print(intensity[:5])