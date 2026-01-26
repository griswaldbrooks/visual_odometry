# Visual Odometry

A C++20 visual odometry pipeline with ORB and learned feature matchers (LightGlue), plus Python visualization tools.

## Quick Start

```bash
# Install dependencies and build
pixi run configure
pixi run build

# Download test data (~500MB, 5-10 min)
./scripts/download_tum.sh fr1_xyz

# Run visual odometry
./build/dev/visual_odometry \
    --images data/tum/rgbd_dataset_freiburg1_xyz/rgb \
    --camera data/tum_camera.yaml

# Visualize results
pixi run -e viz python scripts/plot_trajectory.py trajectory.json \
    --images data/tum/rgbd_dataset_freiburg1_xyz/rgb
```

## Installation

This project uses [Pixi](https://pixi.sh) for dependency management.

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Clone and build
git clone <repo-url>
cd visual_odometry
pixi run configure
pixi run build
```

## Downloading Test Data

### TUM RGB-D Dataset (Recommended)

Small sequences for quick testing:

```bash
# Download fr1_xyz (0.47 GB, recommended)
./scripts/download_tum.sh fr1_xyz

# Other options:
./scripts/download_tum.sh fr1_rpy          # 0.42 GB - rotation testing
./scripts/download_tum.sh fr3_nostructure  # 0.21 GB - minimal features
```

### KITTI Dataset

For KITTI, download manually from [cvlibs.net](https://www.cvlibs.net/datasets/kitti/eval_odometry.php):

- Download "odometry data set (grayscale)"
- Extract to `data/kitti/sequences/00/`

## Running Visual Odometry

### Basic Usage (ORB Features)

```bash
./build/dev/visual_odometry \
    --images data/tum/rgbd_dataset_freiburg1_xyz/rgb \
    --camera data/tum_camera.yaml \
    --output trajectory.json
```

### With LightGlue (Learned Features)

Requires ONNX model in `models/disk_lightglue_end2end.onnx`:

```bash
./build/dev/visual_odometry \
    --images data/tum/rgbd_dataset_freiburg1_xyz/rgb \
    --camera data/tum_camera.yaml \
    --matcher lightglue
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--images <dir>` | Image directory (required) | - |
| `--camera <file>` | Camera intrinsics YAML | `data/kitti_camera.yaml` |
| `--output <file>` | Output trajectory JSON | `trajectory.json` |
| `--matcher <name>` | Feature matcher: `orb`, `lightglue` | `orb` |
| `--max-frames <n>` | Limit frames to process | all |

## Visualization Tools

All visualization tools are in `scripts/` and use the `viz` environment:

### Plot Single Trajectory

Interactive visualization with X/Y/Z vs time plots, 3D view, and synced image display:

```bash
pixi run -e viz python scripts/plot_trajectory.py trajectory.json

# With corresponding images
pixi run -e viz python scripts/plot_trajectory.py \
    data/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt \
    --images data/tum/rgbd_dataset_freiburg1_xyz/rgb
```

Features:

- Time scrubber synced across all views
- 3D trajectory with start/end markers
- Image display (when `--images` provided)
- Coverage bar showing where images exist
- Keyboard controls: Arrow keys, Home/End

![Trajectory Visualizer](docs/groundtruth_visualizer1.png)

### Compare Two Trajectories

Compare estimated vs ground truth with time-matched error metrics:

```bash
pixi run -e viz python scripts/compare_trajectories.py \
    --estimated trajectory.json \
    --ground-truth data/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt
```

### 3D Visualization (Viser)

Web-based 3D visualization:

```bash
pixi run -e viz python scripts/visualize_trajectory.py \
    --trajectory trajectory.json \
    --ground-truth data/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt
```

### Generate Test Trajectories

Create synthetic trajectories for testing visualization:

```bash
# Generate shapes: line, circle, rectangle, sinewave
pixi run -e viz python scripts/generate_test_trajectory.py --shape circle > /tmp/circle.txt
pixi run -e viz python scripts/generate_test_trajectory.py --shape sinewave --poses 200 > /tmp/sine.txt

# Visualize
pixi run -e viz python scripts/plot_trajectory.py /tmp/circle.txt
```

## Development

### Build Commands

```bash
pixi run configure    # CMake configure
pixi run build        # Build
pixi run test         # Run tests
pixi run clean        # Clean build
```

### Code Quality

```bash
pixi run format-check  # Check clang-format
pixi run tidy          # Run clang-tidy
pre-commit run -a      # All pre-commit hooks
```

### Code Style

This project follows strict C++20 conventions:

- **East const**: `int const x` not `const int x`
- **snake_case**: All identifiers including types
- **struct over class**: Always use `struct`
- **tl::expected**: For error handling (no exceptions)
- **[[nodiscard]]**: On all value-returning functions

See `.claude/claude.md` for complete style guide.

## Project Structure

```text
visual_odometry/
├── include/visual_odometry/  # C++ headers
├── src/                      # C++ implementation
├── tests/                    # C++ unit tests
├── scripts/                  # Python tools
│   ├── plot_trajectory.py        # Single trajectory visualization
│   ├── compare_trajectories.py   # Two-trajectory comparison
│   ├── visualize_trajectory.py   # 3D viser visualization
│   ├── generate_test_trajectory.py  # Synthetic trajectory generator
│   ├── trajectory_utils.py       # Trajectory loading/interpolation
│   └── download_tum.sh           # TUM dataset downloader
├── data/                     # Test data and camera configs
│   ├── kitti_camera.yaml
│   ├── tum_camera.yaml
│   └── tum/                  # Downloaded TUM sequences
├── models/                   # ONNX models (LightGlue, etc.)
└── pixi.toml                 # Pixi configuration
```

## License

See [LICENSE](LICENSE) for details.
