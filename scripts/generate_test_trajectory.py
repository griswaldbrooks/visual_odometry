#!/usr/bin/env python3
"""Generate synthetic test trajectories in TUM ground truth format.

Used to validate visualization pipeline with known trajectory shapes.

Usage:
    python scripts/generate_test_trajectory.py --shape circle > test_circle.txt
    python scripts/generate_test_trajectory.py --shape line --poses 50
    python scripts/generate_test_trajectory.py --shape rectangle --scale 2.0
    python scripts/generate_test_trajectory.py --shape sinewave --poses 200
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation


def direction_to_quaternion(direction: np.ndarray) -> np.ndarray:
    """Convert a direction vector to a quaternion facing that direction.

    Assumes Z-forward, Y-up camera convention.
    Returns quaternion as [qx, qy, qz, qw].
    """
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    # Default forward is +X, we want to rotate to face 'direction'
    forward = np.array([1.0, 0.0, 0.0])

    if np.allclose(direction, forward):
        return np.array([0.0, 0.0, 0.0, 1.0])
    elif np.allclose(direction, -forward):
        # 180 degree rotation around Y
        return np.array([0.0, 1.0, 0.0, 0.0])

    # Rotation axis is cross product
    axis = np.cross(forward, direction)
    axis = axis / np.linalg.norm(axis)

    # Rotation angle
    angle = np.arccos(np.clip(np.dot(forward, direction), -1.0, 1.0))

    rot = Rotation.from_rotvec(axis * angle)
    return rot.as_quat()  # [qx, qy, qz, qw]


def generate_line(num_poses: int, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate straight line trajectory along X axis.

    Returns: (timestamps, positions, quaternions)
    """
    timestamps = np.linspace(0.0, num_poses / 10.0, num_poses)

    positions = np.zeros((num_poses, 3))
    positions[:, 0] = np.linspace(0.0, scale * 10.0, num_poses)  # X increases

    # All facing +X direction
    quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (num_poses, 1))

    return timestamps, positions, quaternions


def generate_circle(num_poses: int, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate circular trajectory in XY plane.

    Returns: (timestamps, positions, quaternions)
    """
    timestamps = np.linspace(0.0, num_poses / 10.0, num_poses)

    radius = scale * 5.0
    angles = np.linspace(0.0, 2 * np.pi, num_poses, endpoint=False)

    positions = np.zeros((num_poses, 3))
    positions[:, 0] = radius * np.cos(angles)
    positions[:, 1] = radius * np.sin(angles)

    # Quaternions facing tangent direction (direction of travel)
    quaternions = np.zeros((num_poses, 4))
    for i, angle in enumerate(angles):
        # Tangent direction (derivative of circle)
        tangent = np.array([-np.sin(angle), np.cos(angle), 0.0])
        quaternions[i] = direction_to_quaternion(tangent)

    return timestamps, positions, quaternions


def generate_rectangle(num_poses: int, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate rectangular trajectory in XY plane.

    Returns: (timestamps, positions, quaternions)
    """
    timestamps = np.linspace(0.0, num_poses / 10.0, num_poses)

    width = scale * 8.0
    height = scale * 4.0

    # Distribute poses evenly along perimeter
    perimeter = 2 * (width + height)
    poses_per_unit = num_poses / perimeter

    positions = []
    directions = []

    # Bottom edge (left to right)
    n1 = max(1, int(width * poses_per_unit))
    for x in np.linspace(0, width, n1, endpoint=False):
        positions.append([x, 0, 0])
        directions.append([1, 0, 0])

    # Right edge (bottom to top)
    n2 = max(1, int(height * poses_per_unit))
    for y in np.linspace(0, height, n2, endpoint=False):
        positions.append([width, y, 0])
        directions.append([0, 1, 0])

    # Top edge (right to left)
    n3 = max(1, int(width * poses_per_unit))
    for x in np.linspace(width, 0, n3, endpoint=False):
        positions.append([x, height, 0])
        directions.append([-1, 0, 0])

    # Left edge (top to bottom)
    n4 = max(1, int(height * poses_per_unit))
    for y in np.linspace(height, 0, n4, endpoint=False):
        positions.append([0, y, 0])
        directions.append([0, -1, 0])

    # Resample to exact num_poses
    positions = np.array(positions)
    directions = np.array(directions)

    if len(positions) != num_poses:
        # Interpolate to get exact number
        old_t = np.linspace(0, 1, len(positions))
        new_t = np.linspace(0, 1, num_poses)
        new_positions = np.zeros((num_poses, 3))
        for i in range(3):
            new_positions[:, i] = np.interp(new_t, old_t, positions[:, i])
        positions = new_positions

        # Recompute directions from position differences
        directions = np.zeros((num_poses, 3))
        for i in range(num_poses):
            if i < num_poses - 1:
                directions[i] = positions[i + 1] - positions[i]
            else:
                directions[i] = positions[i] - positions[i - 1]
            norm = np.linalg.norm(directions[i])
            if norm > 1e-10:
                directions[i] /= norm
            else:
                directions[i] = [1, 0, 0]

    quaternions = np.array([direction_to_quaternion(d) for d in directions])

    return timestamps, positions, quaternions


def generate_sinewave(num_poses: int, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sinusoidal trajectory (X forward, Y oscillating).

    Returns: (timestamps, positions, quaternions)
    """
    timestamps = np.linspace(0.0, num_poses / 10.0, num_poses)

    positions = np.zeros((num_poses, 3))
    positions[:, 0] = np.linspace(0.0, scale * 10.0, num_poses)  # X increases
    positions[:, 1] = scale * np.sin(positions[:, 0] * 0.5) * 2.0  # Y oscillates

    # Compute directions from derivatives
    # dx/dt = const, dy/dt = scale * cos(x * 0.5) * 0.5 * 2.0
    quaternions = np.zeros((num_poses, 4))
    for i in range(num_poses):
        x = positions[i, 0]
        dx = scale * 10.0 / num_poses
        dy = scale * np.cos(x * 0.5) * 0.5 * 2.0 * dx
        direction = np.array([dx, dy, 0.0])
        quaternions[i] = direction_to_quaternion(direction)

    return timestamps, positions, quaternions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic test trajectories in TUM format"
    )
    parser.add_argument(
        "--shape",
        type=str,
        choices=["line", "circle", "rectangle", "sinewave"],
        default="circle",
        help="Trajectory shape to generate",
    )
    parser.add_argument(
        "--poses",
        type=int,
        default=100,
        help="Number of poses to generate",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for trajectory size",
    )
    args = parser.parse_args()

    # Generate trajectory
    generators = {
        "line": generate_line,
        "circle": generate_circle,
        "rectangle": generate_rectangle,
        "sinewave": generate_sinewave,
    }

    timestamps, positions, quaternions = generators[args.shape](args.poses, args.scale)

    # Output header
    print(f"# Test trajectory: {args.shape}")
    print(f"# Poses: {args.poses}, Scale: {args.scale}")
    print("# Format: timestamp tx ty tz qx qy qz qw")

    # Output poses in TUM format
    for i in range(len(timestamps)):
        t = timestamps[i]
        x, y, z = positions[i]
        qx, qy, qz, qw = quaternions[i]
        print(f"{t:.6f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")


if __name__ == "__main__":
    main()
