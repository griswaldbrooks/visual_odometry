# Test Data

This directory contains test data for the visual odometry system.

## KITTI Odometry Dataset

Download the KITTI odometry dataset from:
https://www.cvlibs.net/datasets/kitti/eval_odometry.php

### Quick Start

1. Download "odometry data set (grayscale, 22 GB)" or just sequence 00
2. Extract to `data/kitti/sequences/00/`
3. Structure should be:
   ```
   data/kitti/sequences/00/
   ├── image_0/       # Left grayscale camera
   │   ├── 000000.png
   │   ├── 000001.png
   │   └── ...
   ├── image_1/       # Right grayscale camera
   ├── calib.txt
   └── times.txt
   ```

### Camera Intrinsics (Sequence 00, Camera 0)

```
fx = 718.856
fy = 718.856
cx = 607.1928
cy = 185.2157
```

These are also provided in `kitti_camera.yaml`.

### Ground Truth

Download poses from the "odometry ground truth poses (4 MB)" link.
Extract to `data/kitti/poses/00.txt`.

Format: Each line is a 3x4 transformation matrix (row-major):
```
r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
```
