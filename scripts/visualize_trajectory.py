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
        return json.load(f)


def load_kitti_poses(filepath: Path) -> list[np.ndarray]:
    """Load KITTI ground truth poses (3x4 matrices)."""
    poses = []
    with open(filepath) as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            pose = np.array(values).reshape(3, 4)
            poses.append(pose)
    return poses


def pose_to_position_quaternion(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 3x4 pose matrix to position and quaternion (wxyz)."""
    position = pose[:, 3]
    rotation = Rotation.from_matrix(pose[:, :3])
    quat = rotation.as_quat()  # xyzw
    wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
    return position, wxyz


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
        help="Path to KITTI ground truth poses",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum frames to visualize",
    )
    args = parser.parse_args()

    server = viser.ViserServer()
    print(f"Viser server started at: http://localhost:{server.port}")

    # Add world frame
    server.scene.add_frame("world", wxyz=(1, 0, 0, 0), position=(0, 0, 0), axes_length=1.0)

    # Load and visualize ground truth if available
    gt_path = args.ground_truth
    if gt_path.exists():
        print(f"Loading ground truth from {gt_path}")
        gt_poses = load_kitti_poses(gt_path)[: args.max_frames]

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
