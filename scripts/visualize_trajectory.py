#!/usr/bin/env python3
"""Visualize visual odometry trajectory using viser."""

import argparse
import json
import numpy as np
import viser
from pathlib import Path
from scipy.spatial.transform import Rotation


def load_trajectory_json(filepath: Path) -> list[dict]:
    """Load trajectory from JSON file output by C++ VO."""
    with open(filepath) as f:
        data = json.load(f)
    # Handle both {"poses": [...]} and [...] formats
    if isinstance(data, dict) and "poses" in data:
        return data["poses"]
    return data


def load_kitti_poses(filepath: Path) -> list[np.ndarray]:
    """Load KITTI ground truth poses (3x4 matrices)."""
    poses = []
    with open(filepath) as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            pose = np.array(values).reshape(3, 4)
            poses.append(pose)
    return poses


def load_tum_poses(filepath: Path) -> list[np.ndarray]:
    """Load TUM ground truth poses (timestamp tx ty tz qx qy qz qw).

    Returns list of 3x4 pose matrices for compatibility with KITTI loader.
    """
    poses = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            values = [float(x) for x in line.split()]
            # TUM format: timestamp tx ty tz qx qy qz qw
            if len(values) != 8:
                continue
            _, tx, ty, tz, qx, qy, qz, qw = values
            # Convert quaternion to rotation matrix
            rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            # Build 3x4 pose matrix
            pose = np.zeros((3, 4))
            pose[:, :3] = rot
            pose[:, 3] = [tx, ty, tz]
            poses.append(pose)
    return poses


def detect_and_load_poses(filepath: Path) -> list[np.ndarray]:
    """Auto-detect ground truth format (KITTI or TUM) and load poses."""
    with open(filepath) as f:
        first_line = f.readline().strip()
        # TUM files start with # comment or have 8 values (timestamp + pose)
        if first_line.startswith("#"):
            return load_tum_poses(filepath)
        values = first_line.split()
        if len(values) == 8:
            return load_tum_poses(filepath)
        elif len(values) == 12:
            return load_kitti_poses(filepath)
        else:
            raise ValueError(f"Unknown ground truth format: {len(values)} values per line")


def pose_to_position_quaternion(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 3x4 pose matrix to position and quaternion (wxyz)."""
    position = pose[:, 3]
    rotation = Rotation.from_matrix(pose[:, :3])
    quat = rotation.as_quat()  # xyzw
    wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
    return position, wxyz


def compute_ate(gt_positions: np.ndarray, est_positions: np.ndarray) -> dict:
    """Compute Absolute Trajectory Error (ATE) metrics.

    Args:
        gt_positions: Ground truth positions (N, 3)
        est_positions: Estimated positions (N, 3)

    Returns:
        Dictionary with RMSE, mean, median, std, min, max errors
    """
    n = min(len(gt_positions), len(est_positions))
    gt = np.array(gt_positions[:n])
    est = np.array(est_positions[:n])

    # Compute per-frame translation errors
    errors = np.linalg.norm(gt - est, axis=1)

    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "num_frames": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize VO trajectory")
    parser.add_argument(
        "--trajectory",
        type=Path,
        help="Path to trajectory JSON from C++ VO",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/kitti/poses/00.txt"),
        help="Path to ground truth poses (auto-detects KITTI or TUM format)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum frames to visualize",
    )
    args = parser.parse_args()

    server = viser.ViserServer()
    print(f"Viser server started at: http://localhost:{server.get_port()}")

    # Add world frame
    server.scene.add_frame("world", wxyz=(1, 0, 0, 0), position=(0, 0, 0), axes_length=1.0)

    # Load and visualize ground truth if available
    gt_path = args.ground_truth
    gt_positions = []
    if gt_path.exists():
        print(f"Loading ground truth from {gt_path}")
        gt_poses = detect_and_load_poses(gt_path)[: args.max_frames]

        gt_positions = []
        for i, pose in enumerate(gt_poses):
            pos, quat = pose_to_position_quaternion(pose)
            gt_positions.append(pos)

            # Add camera frustum every 10 frames
            if i % 10 == 0:
                server.scene.add_camera_frustum(
                    f"gt/camera_{i:04d}",
                    fov=60.0,
                    aspect=1241 / 376,
                    scale=0.2,
                    wxyz=tuple(quat),
                    position=tuple(pos),
                    color=(0, 255, 0),  # Green for ground truth
                )

        # Add ground truth trajectory line
        if len(gt_positions) > 1:
            server.scene.add_spline_catmull_rom(
                "gt/trajectory",
                np.array(gt_positions),
                color=(0, 255, 0),
            )
        print(f"  Added {len(gt_poses)} ground truth poses")

    # Load and visualize estimated trajectory if provided
    if args.trajectory and args.trajectory.exists():
        print(f"Loading estimated trajectory from {args.trajectory}")
        trajectory = load_trajectory_json(args.trajectory)

        est_positions = []
        for i, frame in enumerate(trajectory[: args.max_frames]):
            pos = np.array(frame["translation"])
            rot = np.array(frame["rotation"]).reshape(3, 3)
            quat = Rotation.from_matrix(rot).as_quat()
            wxyz = (quat[3], quat[0], quat[1], quat[2])
            est_positions.append(pos)

            # Add camera frustum every 10 frames
            if i % 10 == 0:
                server.scene.add_camera_frustum(
                    f"est/camera_{i:04d}",
                    fov=60.0,
                    aspect=1241 / 376,
                    scale=0.2,
                    wxyz=wxyz,
                    position=tuple(pos),
                    color=(255, 0, 0),  # Red for estimated
                )

        # Add estimated trajectory line
        if len(est_positions) > 1:
            server.scene.add_spline_catmull_rom(
                "est/trajectory",
                np.array(est_positions),
                color=(255, 0, 0),
            )
        print(f"  Added {len(trajectory)} estimated poses")

        # Compute ATE if ground truth is available
        if gt_path.exists() and len(gt_positions) > 0:
            ate = compute_ate(gt_positions, est_positions)
            print(f"\n  Absolute Trajectory Error (ATE):")
            print(f"    RMSE:   {ate['rmse']:.3f} m")
            print(f"    Mean:   {ate['mean']:.3f} m")
            print(f"    Median: {ate['median']:.3f} m")
            print(f"    Std:    {ate['std']:.3f} m")
            print(f"    Min:    {ate['min']:.3f} m")
            print(f"    Max:    {ate['max']:.3f} m")
            print(f"    Frames: {ate['num_frames']}")

    print("\nOpen the URL above in your browser to see the visualization.")
    print("  Green = Ground Truth")
    print("  Red = Estimated")
    print("\nPress Ctrl+C to exit.")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
