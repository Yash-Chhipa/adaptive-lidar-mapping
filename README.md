# Adaptive LiDAR Mapping

> A variable-resolution 3D LiDAR perception and spatial mapping pipeline for autonomous ground vehicles and robotic platforms.

Adaptive LiDAR Mapping is a Python-based perception and mapping system designed to process 3D LiDAR point clouds, separate terrain from obstacles, identify spatially distinct objects, track motion in synthetic sequences, and convert the point cloud into a **range-dependent variable-resolution spatial grid**.

The central idea is simple:

> **Use high spatial resolution where precision matters most, and progressively reduce resolution with distance.**

The system currently supports both **synthetic LiDAR data** and **real KITTI Velodyne LiDAR data**.

---

## 📌 Project Overview

Traditional uniform-resolution occupancy or elevation grids can become extremely expensive as the sensing area and required resolution increase.

For example, representing a:

```text
200 m × 200 m
```

environment entirely at:

```text
5 cm × 5 cm
```

resolution requires:

```text
4000 × 4000 = 16,000,000 cells
```

Most of these cells may contain little or no useful information.

Adaptive LiDAR Mapping addresses this by changing the grid resolution according to the distance from the LiDAR sensor.

```text
                         LiDAR Sensor
                              ●
                              │
               ┌──────────────┼──────────────┐
               │              │              │
            0–10 m         10–50 m        50–100 m
               │              │              │
              5 cm           20 cm           50 cm
           resolution      resolution      resolution
```

This produces a sparse, range-aware spatial representation that can serve as a foundation for downstream navigation, obstacle avoidance, and autonomous path planning.

---

## ✨ Features

### 1. Synthetic LiDAR Simulation

The project includes a deterministic synthetic LiDAR environment for development, testing, and demonstration.

The synthetic scene contains:

- Flat ground
- Multiple static obstacles
- A moving pedestrian
- 3D point-cloud measurements
- Frame-to-frame movement

The synthetic generator allows the perception pipeline to be tested without requiring external datasets.

It also provides a controlled environment for demonstrating dynamic-object detection and temporal tracking.

---

### 2. Real KITTI LiDAR Support

The project supports real LiDAR data from the **KITTI Vision Benchmark Suite**.

The KITTI loader reads Velodyne `.bin` files containing:

```text
X
Y
Z
Intensity
```

for each LiDAR point.

The loader converts the raw binary data into NumPy arrays that can then be processed by the perception pipeline.

### ⚠️ KITTI Data Is Not Included

The KITTI dataset is **not included in this repository**.

If you want to use KITTI mode locally, you must:

1. Obtain the appropriate KITTI dataset yourself.
2. Download the required Velodyne LiDAR `.bin` files.
3. Place the data in the directory structure expected by the application.
4. Run the project locally.

The KITTI dataset is an external dataset and is not distributed with this project.

---

## ⚠️ Deployed Version Limitation

The currently deployed Streamlit application is configured to demonstrate the project using **Synthetic Data Mode**.

### Deployed Application

```text
Synthetic Mode
      ↓
Works directly
      ↓
No external dataset required
```

### KITTI Mode

```text
KITTI Mode
      ↓
Requires KITTI dataset
      ↓
Dataset is not included in deployment
```

Therefore:

> **Use Synthetic Mode when accessing the deployed application.**

The deployed version does not contain the KITTI dataset.

If you want to experiment with real KITTI LiDAR data, clone the repository, download the required KITTI data yourself, configure the expected local data path, and run the application locally.

This separation is intentional.

---

# 🧠 Perception Pipeline

The complete perception pipeline is:

```text
                     LiDAR Data
                         │
              ┌──────────┴──────────┐
              │                     │
          Synthetic               KITTI
              │                     │
              └──────────┬──────────┘
                         ↓
                  Point Cloud
                         │
                         ↓
              Ground Segmentation
                  RANSAC Plane
                         │
                ┌────────┴────────┐
                │                 │
              Ground          Non-Ground
                                  │
                                  ↓
                         DBSCAN Clustering
                                  │
                                  ↓
                          Object Candidates
                                  │
                     ┌────────────┴────────────┐
                     │                         │
                  Static                    Dynamic
                     │                         │
                     └────────────┬────────────┘
                                  ↓
                       Adaptive Grid Mapping
                                  │
                                  ↓
                      Spatial Representation
                                  │
                                  ↓
                       Streamlit Visualization
```

---

# 🌍 Ground Segmentation

Ground segmentation is performed using **RANSAC plane fitting** through Open3D.

The algorithm searches for a dominant planar surface representing the ground.

The point cloud is separated into:

```text
Ground Points
      +
Non-Ground Points
```

The non-ground points are then passed to the object clustering stage.

RANSAC parameters can be adjusted through the dashboard.

---

# 🎯 Object Clustering

Non-ground points are grouped using **DBSCAN clustering**.

DBSCAN is useful for point clouds because it can group spatially dense points without requiring the number of objects to be known beforehand.

The clustering stage uses parameters such as:

```text
eps
min_points
```

Noise points that do not belong to meaningful clusters can be excluded.

The resulting clusters represent detected spatial obstacles or objects.

---

# 🚶 Static and Dynamic Objects

The synthetic pipeline supports dynamic-object identification using inter-frame centroid movement.

The process is approximately:

```text
Frame N
   ↓
Object Centroids
   ↓
Frame N+1
   ↓
New Object Centroids
   ↓
Centroid Displacement
   ↓
Movement Threshold
   ↓
Static / Dynamic Classification
```

The synthetic environment contains a moving pedestrian, allowing the system to demonstrate dynamic-object detection across consecutive frames.

---

# 📍 Temporal Tracking

The project also contains a centroid-based temporal tracking component.

Tracks maintain information such as:

- Track ID
- Position
- Velocity
- Speed
- Object state
- Recent trajectory
- Missed frames

The current implementation uses **nearest-centroid matching** for associating detections between frames.

It is not currently based on a Kalman filter or Hungarian assignment algorithm.

---

# 🗺️ Variable-Resolution Grid

The core component of the project is the adaptive spatial grid.

The current configuration is:

| Distance Range | Cell Size |
|---|---:|
| 0–10 m | 0.05 m |
| 10–50 m | 0.20 m |
| 50–100 m | 0.50 m |

For every LiDAR point:

```text
(x, y, z)
   ↓
Calculate planar distance
r = √(x² + y²)
   ↓
Select resolution tier
   ↓
Calculate grid cell
   ↓
Store / update cell
```

Each occupied cell stores:

- Maximum observed height
- Object label
- Number of points

The grid therefore provides a compact spatial representation of the observed environment.

---

# 📊 Memory Reduction

A uniform 5 cm grid over a 200 m × 200 m area requires:

```text
4000 × 4000
=
16,000,000 cells
```

The adaptive grid only creates cells that are actually occupied by LiDAR observations.

During testing, a representative KITTI frame contained approximately:

```text
123,397 LiDAR points
```

The adaptive grid produced approximately:

```text
40,086 occupied cells
```

compared with:

```text
16,000,000 cells
```

for the equivalent uniform 5 cm grid.

This corresponds to approximately:

```text
99.75% reduction
```

in the number of allocated grid cells for that tested frame.

> The exact reduction depends on the point cloud, sensor range, scene geometry, and configured resolution tiers.

---

# 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard for visualizing the perception pipeline.

## Data Sources

- Synthetic LiDAR
- KITTI LiDAR

## Frame Controls

- Frame selection
- Frame stepping
- Reset
- Sequence playback

## Perception Parameters

- RANSAC threshold
- DBSCAN epsilon
- DBSCAN minimum points
- Maximum processing range

## Tracking Parameters

- Matching distance
- Dynamic movement threshold
- Track persistence

## Visualization

The dashboard provides a top-down LiDAR visualization containing:

- Point cloud
- Sensor position
- Range rings
- Object clusters
- Static/dynamic information
- Adaptive grid representation

## Telemetry

The dashboard displays metrics related to:

- Point count
- Object count
- Dynamic objects
- Grid cells
- Memory reduction
- Processing performance

---

# 🛠️ Technology Stack

## Programming Language

- **Python**

## Numerical Computing

- **NumPy**

Used for:

- Point-cloud arrays
- Coordinate manipulation
- Distance calculations
- Grid calculations
- Synthetic data generation

## Point Cloud Processing

- **Open3D**

Used for:

- RANSAC plane segmentation
- DBSCAN clustering
- Point-cloud manipulation

## Dashboard & Visualization

- **Streamlit**
- **Plotly**
- HTML/CSS customization

## Data Sources

- Synthetic LiDAR point clouds
- KITTI Velodyne `.bin` files

---

# 📁 Project Structure

```text
adaptive-lidar-mapping/
│
├── app_re.py
│   └── Main Streamlit dashboard
│
├── app.py
│   └── Earlier application version
│
├── data_gen.py
│   └── Synthetic LiDAR scene generator
│
├── kitti_loader.py
│   └── KITTI Velodyne .bin loader
│
├── terrain_segmentation.py
│   └── RANSAC ground segmentation
│
├── object_clustering.py
│   └── DBSCAN clustering and object classification
│
├── tracking.py
│   └── Temporal centroid-based tracking
│
├── variable_resolution_grid.py
│   └── Adaptive multi-tier spatial grid
│
├── kitti_benchmark.py
│   └── KITTI processing benchmark
│
├── test_kitti.py
│   └── KITTI pipeline testing
│
├── test_kitti_grid.py
│   └── KITTI adaptive-grid testing
│
├── test_loader.py
│   └── LiDAR loader testing
│
├── test_pipeline.py
│   └── End-to-end synthetic pipeline testing
│
├── requirements.txt
│   └── Python dependencies
│
└── .gitignore
    └── Excludes local datasets and unnecessary files
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Yash-Chhipa/adaptive-lidar-mapping.git
cd adaptive-lidar-mapping
```

---

## 2. Create a Virtual Environment

Using a dedicated virtual environment is recommended.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The primary dependencies are:

```text
numpy
open3d
streamlit
plotly
```

---

# ▶️ Running the Application

Start the Streamlit dashboard with:

```bash
streamlit run app_re.py
```

The application will open in your browser.

If it does not open automatically, Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

---

# 🧪 Running Synthetic Mode

Synthetic mode requires **no external dataset**.

Start the application:

```bash
streamlit run app_re.py
```

Then select:

```text
Data Source → Synthetic
```

The application will generate the LiDAR scene and process it through the perception pipeline.

---

# 🚗 Running KITTI Mode Locally

KITTI mode requires you to obtain the dataset independently.

After downloading the appropriate KITTI Velodyne data, configure the local dataset path expected by the project.

A general structure may look like:

```text
adaptive-lidar-mapping/
│
├── data/
│   └── kitti/
│       └── ...
│           └── *.bin
│
├── app_re.py
├── kitti_loader.py
└── ...
```

The exact dataset structure and path must match the configuration used by the application.

Then run:

```bash
streamlit run app_re.py
```

and select:

```text
Data Source → KITTI
```

### Important

The KITTI dataset is **not included in this GitHub repository**.

You must download and configure the dataset yourself before using KITTI mode locally.

---

# 📈 Example Benchmark

A representative KITTI frame used during development contained:

```text
123,397 LiDAR points
```

The adaptive grid produced:

```text
40,086 occupied cells
```

while the equivalent uniform 5 cm grid would contain:

```text
16,000,000 cells
```

Result:

```text
~99.75% reduction in allocated grid cells
```

This demonstrates the primary motivation behind the adaptive-resolution approach.

---

# 🔬 Current Capabilities

The current MVP supports:

- Synthetic LiDAR generation
- Real KITTI `.bin` ingestion
- RANSAC ground segmentation
- DBSCAN obstacle clustering
- Synthetic dynamic-object detection
- Centroid-based temporal tracking
- Variable-resolution spatial mapping
- Adaptive-grid memory analysis
- Interactive Streamlit visualization
- KITTI frame playback locally
- Configurable perception parameters

---

# ⚠️ Current Limitations

## 1. KITTI Dataset Is Not Included

The GitHub repository does not contain the KITTI dataset.

Local KITTI processing requires the user to download the dataset independently.

---

## 2. Deployed Version Uses Synthetic Mode

The currently deployed Streamlit application is intended to demonstrate the system using **Synthetic Data Mode**.

The deployed application does not include the KITTI dataset.

Therefore:

> **Use Synthetic Mode on the deployed application.**

To experiment with KITTI:

> Download the KITTI data yourself and run the project locally.

---

## 3. KITTI Dynamic Tracking

The current real-KITTI pipeline primarily performs single-frame perception and adaptive-grid generation.

Dynamic-object classification requires temporal information across consecutive frames and is therefore demonstrated primarily through the synthetic sequence pipeline at the current MVP stage.

---

## 4. Classical Perception

The current perception pipeline uses:

```text
RANSAC
   +
DBSCAN
   +
Centroid-Based Tracking
```

It does not currently use a deep-learning LiDAR perception network such as:

- PointNet++
- Cylinder3D
- MinkowskiNet

---

# 🔐 Dataset & Repository Policy

Large LiDAR datasets are intentionally excluded from the repository.

The project repository contains:

```text
Source Code
Configuration
Tests
Requirements
Documentation
```

but does not contain the external KITTI dataset.

This keeps the repository manageable and maintains a clear separation between:

```text
Project Code
```

and:

```text
External Dataset
```

---

# 🚧 Future Development

Potential future extensions include:

- Multi-frame tracking on real KITTI sequences
- Improved data association
- Kalman-filter-based tracking
- Hungarian assignment
- 3D oriented bounding boxes
- Improved elevation mapping
- Traversability / costmap generation
- A* or Hybrid A* path planning integration
- ROS2 integration
- Real-time LiDAR streaming
- Lightweight learned point-cloud segmentation
- Remote KITTI demo-data support for cloud deployment
- Hardware LiDAR integration

---

# 🎯 Intended Applications

The architecture is intended as a foundation for:

- Autonomous ground vehicles
- Robotic platforms
- LiDAR-based navigation
- Obstacle detection
- Terrain perception
- Spatial mapping
- Autonomous path planning
- Resource-efficient perception systems

The adaptive-resolution grid is particularly useful in scenarios where maintaining very high spatial resolution across the entire sensing area would be unnecessarily expensive.

---

# 📚 Technical Summary

At a high level, the project combines:

```text
3D LiDAR
    +
Geometric Perception
    +
Object Clustering
    +
Temporal Tracking
    +
Adaptive Spatial Representation
    +
Interactive Visualization
```

The primary architectural principle is:

> **Allocate computational and spatial resources according to where they provide the most value.**

Instead of representing the entire environment at the finest available resolution, the system prioritizes high-resolution mapping near the sensor and progressively reduces resolution with distance.

---

# 👨‍💻 Team

**404 Developers**

---

# ⭐ Acknowledgements

This project uses the **KITTI Vision Benchmark Suite** for real-world LiDAR experimentation.

KITTI is an external dataset and is **not distributed with this repository**.

Users who wish to reproduce the real-LiDAR experiments should obtain the appropriate KITTI data directly from the official source and comply with its applicable dataset terms.

---

# 🚀 Quick Start

For the fastest way to try the project:

```bash
git clone https://github.com/Yash-Chhipa/adaptive-lidar-mapping.git
cd adaptive-lidar-mapping

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

streamlit run app_re.py
```

Then select:

```text
Synthetic Mode
```

to run the complete demonstration without downloading any external dataset.

For real KITTI LiDAR:

```text
Download KITTI dataset
        ↓
Place/configure the .bin files locally
        ↓
Run app_re.py
        ↓
Select KITTI Mode
```

### In short

| Mode | Local | Deployed |
|---|---|---|
| **Synthetic** | ✅ Works | ✅ Works |
| **KITTI** | ✅ Requires downloaded dataset | ❌ Dataset not included |

> **Synthetic Mode works out of the box. KITTI Mode requires the user to obtain and configure the KITTI dataset separately.**
