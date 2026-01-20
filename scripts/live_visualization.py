#!/usr/bin/env python3
"""Live visual odometry visualization using viser.

This script demonstrates using the Python bindings to run the VO pipeline
and visualize the trajectory in real-time using viser.

Usage:
    python live_visualization.py --images <dir> [--camera <yaml>]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import viser
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description="Live VO visualization")
    parser.add_argument("--images", type=Path, required=True, help="Path to image directory")
    parser.add_argument("--camera", type=Path, default=Path("data/kitti_camera.yaml"),
                        help="Camera intrinsics YAML")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay between frames (seconds)")
    args = parser.parse_args()

    # Import visual_odometry (assumes PYTHONPATH is set)
    import visual_odometry as vo

    print(f"Visual Odometry Live Visualization")
    print(f"  Images: {args.images}")
    print(f"  Camera: {args.camera}")
    print(f"  Version: {vo.__version__}")

    # Load components
    intrinsics = vo.CameraIntrinsics.load_from_yaml(str(args.camera))
    print(f"  Loaded intrinsics: fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}")

    loader = vo.ImageLoader.create(str(args.images))
    print(f"  Found {len(loader)} images")

    matcher = vo.create_matcher("orb")
    estimator = vo.MotionEstimator(intrinsics)
    trajectory = vo.Trajectory()

    # Start viser server
    server = viser.ViserServer()
    print(f"\nViser server: http://localhost:{server.get_port()}")
    print("Open in browser to see live visualization.\n")

    # Add world frame
    server.scene.add_frame("world", wxyz=(1, 0, 0, 0), position=(0, 0, 0), axes_length=1.0)

    # Process frames
    positions = []
    for i, (img1, img2) in enumerate(loader):
        # Match and estimate
        result = matcher.match_images(img1, img2)
        motion = estimator.estimate(result.points1, result.points2)

        if motion:
            trajectory.add_motion(motion)

        # Get current pose
        pose = trajectory.current_pose
        pos = pose.translation
        rot = Rotation.from_matrix(pose.rotation)
        quat = rot.as_quat()  # xyzw
        wxyz = (quat[3], quat[0], quat[1], quat[2])

        positions.append(pos)

        # Add camera frustum
        server.scene.add_camera_frustum(
            f"camera_{i:04d}",
            fov=60.0,
            aspect=1241 / 376,
            scale=0.15,
            wxyz=wxyz,
            position=tuple(pos),
            color=(0, 255, 0) if motion else (255, 0, 0),
        )

        # Update trajectory line
        if len(positions) > 1:
            server.scene.add_spline_catmull_rom(
                "trajectory",
                np.array(positions),
                color=(0, 200, 255),
            )

        # Status
        status = "OK" if motion else "FAILED"
        print(f"  Frame {i+1:4d}/{len(loader)-1} | Matches: {len(result):4d} | "
              f"Inliers: {motion.inliers:3d} | {status}")

        time.sleep(args.delay)

    print(f"\nProcessed {len(trajectory)} poses")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
