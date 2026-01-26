# Visual Odometry Architecture

**Last Updated:** 2026-01-25

This document describes the high-level architecture and algorithmic pipeline of the visual odometry system. For implementation details, coding patterns, and design standards, see [DESIGN_AND_STANDARDS.md](DESIGN_AND_STANDARDS.md).

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Algorithmic Pipeline](#algorithmic-pipeline)
4. [Data Flow](#data-flow)
5. [Available Matchers](#available-matchers)
6. [Future Directions](#future-directions)

---

## System Overview

### High-Level Pipeline

The visual odometry system processes sequential images through a functional pipeline to estimate camera trajectory:

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT - SYSTEM BOUNDARY                                         │
├─────────────────────────────────────────────────────────────────┤
│ 1. Parse command line arguments                                 │
│ 2. Load camera intrinsics from YAML                             │
│ 3. Create image loader (scans directory)                        │
│ 4. Select matching approach:                                    │
│    • Hand-crafted: ORB descriptors + RANSAC matching            │
│    • Learned: LightGlue (with DISK/SuperPoint features)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ PROCESSING LOOP                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   For each consecutive image pair:                             │
│                                                                 │
│   1. Feature matching                                           │
│      Extract and match features between image pairs            │
│           ↓                                                     │
│   2. Motion estimation                                          │
│      Compute relative camera motion (rotation & translation)    │
│      using essential matrix and RANSAC                          │
│           ↓                                                     │
│   3. Trajectory accumulation                                    │
│      Integrate motion into world-frame camera trajectory        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT - SYSTEM BOUNDARY                                        │
├─────────────────────────────────────────────────────────────────┤
│ • Save trajectory to JSON                                       │
│ • Optional: Compare with ground truth                           │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
include/visual_odometry/
├── feature_detector.hpp    # Pure function: detect_features()
├── feature_matcher.hpp     # Pure function: match_features()
├── motion_estimator.hpp    # Pure function: estimate_motion()
├── image_matcher.hpp       # Matcher implementations (ORB, LightGlue)
├── matcher_concept.hpp     # C++20 concept: matcher_like
├── trajectory.hpp          # Stateful pose accumulation
├── image_loader.hpp        # Stateful image iteration
└── onnx_session.hpp        # RAII wrapper for ONNX Runtime

src/
├── feature_detector.cpp    # ORB-based feature detection
├── feature_matcher.cpp     # Brute-force matching + ratio test
├── motion_estimator.cpp    # Essential matrix → R, t recovery
├── image_matcher.cpp       # ORB and LightGlue implementations
├── trajectory.cpp          # Pose accumulation + JSON export
├── image_loader.cpp        # Sequential image loading
├── onnx_session.cpp        # ONNX Runtime inference
├── python_bindings.cpp     # Nanobind Python API
└── main.cpp                # CLI orchestrator

python/visual_odometry/
└── __init__.py             # Python module initialization

scripts/
├── visualize_trajectory.py # Viser-based 3D visualization
├── live_visualization.py   # Real-time VO visualization
└── compare_trajectories.py # Ground truth comparison tools
```

---

## Core Components

### 1. Feature Detection

**Purpose:** Extract distinctive keypoints and descriptors from grayscale images.

**Interface:**
```cpp
[[nodiscard]] auto detect_features(
    cv::Mat const& image,
    feature_detector_config const& config = {})
    -> detection_result;
```

**Algorithm:** Uses OpenCV's ORB (Oriented FAST and Rotated BRIEF)
- FAST corner detection for keypoints
- BRIEF descriptors with rotation invariance
- Configurable number of features (default: 2000)

**Performance:** ~5-10ms per image (1000 features)

---

### 2. Feature Matching

**Purpose:** Find correspondences between keypoints in two images using descriptor similarity.

**Interface:**
```cpp
[[nodiscard]] auto match_features(
    cv::Mat const& desc1,
    cv::Mat const& desc2,
    std::span<cv::KeyPoint const> kp1,
    std::span<cv::KeyPoint const> kp2,
    feature_matcher_config const& config = {})
    -> match_result;
```

**Algorithm:**
- Brute-force matching with Hamming distance (for binary descriptors)
- KNN matching (k=2) for Lowe's ratio test
- Ratio test filters ambiguous matches (default threshold: 0.75)

**Performance:** ~2-5ms per pair (1000 features)

---

### 3. Motion Estimation

**Purpose:** Compute relative camera motion (rotation and translation) from matched point correspondences.

**Interface:**
```cpp
[[nodiscard]] auto estimate_motion(
    std::span<cv::Point2f const> points1,
    std::span<cv::Point2f const> points2,
    camera_intrinsics const& intrinsics,
    motion_estimator_config const& config = {})
    -> motion_estimate;
```

**Algorithm:**
1. Compute essential matrix using RANSAC (`cv::findEssentialMat`)
2. Decompose essential matrix to recover rotation and translation (`cv::recoverPose`)
3. Validate result (minimum inliers, essential matrix validity)

**Output:** Relative pose (R, t) and inlier count, or invalid flag if estimation fails.

**Performance:** ~1-3ms per pair (100-500 matches)

---

### 4. Trajectory

**Purpose:** Accumulate relative motions into absolute world-frame poses.

**Responsibilities:**
- Compose relative transformations into global trajectory
- Maintain full pose history
- Export trajectory in various formats (currently TUM RGB-D JSON)

---

### 5. Image Loader

**Purpose:** Load and iterate through sequential images from a directory.

**Responsibilities:**
- Scan directory for supported image formats (JPG, PNG)
- Load consecutive image pairs
- Convert images to grayscale for processing

---

## Algorithmic Pipeline

### Frame-to-Frame Visual Odometry

The system implements monocular visual odometry using the essential matrix approach:

```
Image Pair (t, t+1)
        ↓
┌──────────────────────┐
│  Feature Detection   │  Extract keypoints and descriptors
│                      │  (ORB or LightGlue's learned features)
└──────────────────────┘
        ↓
┌──────────────────────┐
│  Feature Matching    │  Find correspondences between frames
│                      │  (Descriptor matching or learned matching)
└──────────────────────┘
        ↓
┌──────────────────────┐
│  Motion Estimation   │  Essential matrix + RANSAC
│                      │  → Recover R, t
└──────────────────────┘
        ↓
┌──────────────────────┐
│  Pose Accumulation   │  T_world = T_world * T_relative
│                      │  (SE(3) composition)
└──────────────────────┘
        ↓
    Trajectory
```

### Essential Matrix Geometry

The essential matrix E relates corresponding points in two calibrated cameras:

```
p2^T * E * p1 = 0
```

Where:
- `p1`, `p2` are normalized image coordinates
- `E = [t]_× R` encodes rotation R and translation direction t
- RANSAC finds inliers fitting the epipolar constraint

The essential matrix is decomposed to recover the relative pose, which is then accumulated into the world-frame trajectory.

**Key constraint:** Translation scale is ambiguous (monocular scale drift). The system estimates up to an unknown scale factor.

---

## Data Flow

### Type Transformations Through Pipeline

```
Image Files (PNG/JPG on disk)
    ↓ cv::imread + cvtColor
cv::Mat (CV_8UC1 grayscale)
    ↓ detect_features()
detection_result {
    keypoints: vector<cv::KeyPoint>
    descriptors: cv::Mat (CV_8UC1)
}
    ↓ match_features()
match_result {
    points1: vector<cv::Point2f>
    points2: vector<cv::Point2f>
    matches: vector<cv::DMatch>
}
    ↓ estimate_motion()
motion_estimate {
    rotation: Eigen::Matrix3d
    translation: Eigen::Vector3d
    inliers: int
    valid: bool
}
    ↓ Trajectory::add_motion()
pose {
    rotation: Eigen::Matrix3d
    translation: Eigen::Vector3d
}
[accumulated in poses_ vector]
    ↓ Trajectory::to_json()
JSON string (TUM RGB-D format)
    ↓ write to file
trajectory.json on disk
```

### Coordinate Systems

**Image coordinates** → **Normalized coordinates** → **3D world coordinates**

1. **Image coordinates** (pixels): Raw keypoint positions from feature detection
2. **Normalized coordinates** (metric): Points after applying inverse camera intrinsics
3. **Camera frame**: Right-handed, Z-forward, Y-down (OpenCV convention)
4. **World frame**: Accumulated global poses starting from identity

**Intrinsic calibration** (fx, fy, cx, cy) is required to convert pixel coordinates to normalized coordinates for essential matrix estimation.

---

## Available Matchers

The system supports multiple matching algorithms:

### ORB Matcher (Classical)

**Algorithm:** Hand-crafted pipeline
1. ORB feature detection on both images
2. Brute-force descriptor matching
3. Lowe's ratio test for filtering

**Advantages:**
- No external dependencies beyond OpenCV
- Fast (~10-15ms per pair)
- Lightweight

**Limitations:**
- Lower recall in challenging conditions (lighting, viewpoint changes)
- Fixed feature descriptor

---

### LightGlue Matcher (Learned)

**Algorithm:** Deep learning pipeline (ONNX Runtime)
1. SuperPoint keypoint detection and description
2. LightGlue attention-based matching

**Advantages:**
- Superior matching quality (higher inlier ratios)
- Robust to lighting and viewpoint changes
- State-of-the-art performance on benchmarks

**Requirements:**
- ONNX Runtime dependency
- Pre-exported ONNX model file

**Performance:** ~20-50ms per pair (depends on image size, hardware)

---

## Future Directions

### 1. Python-Driven Main Loop

**Current state:** Main loop in C++ (`main.cpp`)

**Proposed:** Python orchestration for flexibility

Benefits:
- Easy experimentation with different matchers and configurations
- Integration with Python ecosystem (matplotlib, pandas, logging)
- Simpler debugging and visualization hooks
- Keep C++ for performance-critical algorithms

See [DESIGN_AND_STANDARDS.md](DESIGN_AND_STANDARDS.md#python-integration) for examples.

### 2. Real-Time Viser Integration

**Current state:** Post-hoc visualization scripts

**Proposed:** Live trajectory visualization during processing

Benefits:
- Immediate feedback on algorithm performance
- Visual debugging of failures (tracking loss, drift)
- 3D scene understanding

### 3. Additional Learned Matchers

**Candidates:**
- SuperGlue (attention-based matching)
- LoFTR (detector-free matching)
- DKM (Dense Keypoint Matching)
- ASpanFormer (cross-attention)

The polymorphic matcher architecture makes integration straightforward. See [DESIGN_AND_STANDARDS.md](DESIGN_AND_STANDARDS.md#extension-points) for the integration pattern.

### 4. Backend Optimization

**Current:** Frame-to-frame VO (no loop closure)

**Future extensions:**
- Bundle adjustment for global optimization
- Loop closure detection to correct drift
- Pose graph optimization
- Integration with SLAM frameworks (ORB-SLAM3, etc.)

**Design principle:** Keep VO pure, add backend as separate module

### 5. Multi-Camera Support

**Current:** Monocular only

**Extension points:**
- Stereo matching functions for depth estimation
- Multi-camera trajectory fusion
- Extrinsic calibration utilities

---

## Summary

The visual odometry system architecture emphasizes:

**Algorithmic clarity:**
- Pure functions for core algorithms (detection, matching, motion estimation)
- Explicit data flow through well-defined types
- Essential matrix approach for camera motion estimation

**System organization:**
- State isolated to system boundaries (I/O, accumulation, resources)
- Polymorphic matchers via concepts and variants
- Functional core with imperative shell

**Extensibility:**
- Easy integration of new matching algorithms
- Configurable parameters via structs
- Multiple output formats
- Python bindings for high-level orchestration

For implementation details, coding patterns, and extension guides, see [DESIGN_AND_STANDARDS.md](DESIGN_AND_STANDARDS.md).
