# TUM RGB-D Dataset

Small dataset sequences for quick visual odometry testing.

## Quick Start

```bash
# Download fr1_xyz (recommended, 0.47 GB)
./scripts/download_tum.sh fr1_xyz

# Run visual odometry
./build/dev/visual_odometry \
    --images data/tum/rgbd_dataset_freiburg1_xyz/rgb \
    --camera data/tum_camera.yaml
```

## Available Sequences

| Sequence | Size | Duration | Description |
|----------|------|----------|-------------|
| `fr3_nostructure` | 0.21 GB | 16s | Minimal features (challenging) |
| `fr1_rpy` | 0.42 GB | 28s | Rotation along principal axes |
| `fr1_xyz` | 0.47 GB | 30s | Translation along axes (recommended) |
| `fr1_desk` | 0.58 GB | 23s | Office desk scene |

## Download

```bash
./scripts/download_tum.sh <sequence>

# Examples:
./scripts/download_tum.sh fr1_xyz
./scripts/download_tum.sh fr1_rpy
```

## Directory Structure

After download:
```
data/tum/rgbd_dataset_freiburg1_xyz/
├── rgb/                    # Color images (use these)
│   ├── 1305031102.175304.png
│   ├── 1305031102.211214.png
│   └── ...
├── depth/                  # Depth images (not used)
├── rgb.txt                 # Timestamps
├── depth.txt
├── groundtruth.txt         # Ground truth poses
└── accelerometer.txt       # IMU data
```

## Camera Intrinsics

Freiburg1 sequences use `data/tum_camera.yaml`:
- fx: 517.3, fy: 516.5
- cx: 318.6, cy: 255.3
- Resolution: 640x480

## Ground Truth

The `groundtruth.txt` file contains poses in TUM format:
```
timestamp tx ty tz qx qy qz qw
```

## More Info

- [TUM RGB-D Dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
- [Download Page](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)
