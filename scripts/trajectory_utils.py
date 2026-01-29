#!/usr/bin/env python3
"""Trajectory loading and interpolation utilities.

Supports TUM, KITTI, and trajectory.json formats with proper time-aligned
interpolation using linear position interpolation and quaternion slerp.
"""

import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
from dataclasses import dataclass
from typing import Optional


@dataclass
class Pose:
    """A single pose with position and orientation."""

    position: np.ndarray  # (3,) [x, y, z]
    quaternion: np.ndarray  # (4,) [qx, qy, qz, qw]

    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix."""
        mat = np.eye(4)
        mat[:3, :3] = Rotation.from_quat(self.quaternion).as_matrix()
        mat[:3, 3] = self.position
        return mat


class TrajectoryInterpolator:
    """Interpolate trajectory poses at arbitrary timestamps.

    Uses linear interpolation for positions and spherical linear
    interpolation (slerp) for quaternion orientations.
    """

    def __init__(
        self,
        timestamps: np.ndarray,
        positions: np.ndarray,
        quaternions: np.ndarray,
    ):
        """Initialize interpolator.

        Args:
            timestamps: (N,) array of times in seconds
            positions: (N, 3) array of [x, y, z] positions
            quaternions: (N, 4) array of [qx, qy, qz, qw] quaternions
        """
        self.timestamps = np.asarray(timestamps)
        self.positions = np.asarray(positions)
        self.quaternions = np.asarray(quaternions)

        if len(self.timestamps) == 0:
            raise ValueError("Trajectory must have at least one pose")

        if len(self.timestamps) != len(self.positions):
            raise ValueError("timestamps and positions must have same length")

        if len(self.timestamps) != len(self.quaternions):
            raise ValueError("timestamps and quaternions must have same length")

        # Precompute rotations for slerp
        self._rotations = Rotation.from_quat(self.quaternions)

        # Create slerp interpolator if more than one pose
        if len(self.timestamps) > 1:
            self._slerp = Slerp(self.timestamps, self._rotations)
        else:
            self._slerp = None

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        """Total duration of trajectory in seconds."""
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def start_time(self) -> float:
        return float(self.timestamps[0])

    @property
    def end_time(self) -> float:
        return float(self.timestamps[-1])

    def interpolate(self, t: float) -> Pose:
        """Interpolate pose at time t.

        Args:
            t: Query timestamp

        Returns:
            Interpolated Pose at time t

        Notes:
            - If t < start_time, returns first pose
            - If t > end_time, returns last pose
            - Single-pose trajectories always return that pose
        """
        # Handle edge cases
        if len(self.timestamps) == 1:
            return Pose(self.positions[0].copy(), self.quaternions[0].copy())

        if t <= self.timestamps[0]:
            return Pose(self.positions[0].copy(), self.quaternions[0].copy())

        if t >= self.timestamps[-1]:
            return Pose(self.positions[-1].copy(), self.quaternions[-1].copy())

        # Linear interpolation for position
        position = np.array(
            [
                np.interp(t, self.timestamps, self.positions[:, i])
                for i in range(3)
            ]
        )

        # Slerp for quaternion
        rotation = self._slerp([t])[0]
        quaternion = rotation.as_quat()

        return Pose(position, quaternion)

    def interpolate_batch(
        self, query_times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate poses at multiple timestamps.

        Args:
            query_times: (M,) array of query timestamps

        Returns:
            Tuple of (positions, quaternions) arrays, each of shape (M, 3) and (M, 4)
        """
        query_times = np.asarray(query_times)
        n = len(query_times)

        positions = np.zeros((n, 3))
        quaternions = np.zeros((n, 4))

        for i, t in enumerate(query_times):
            pose = self.interpolate(t)
            positions[i] = pose.position
            quaternions[i] = pose.quaternion

        return positions, quaternions


def load_tum_trajectory(path: Path) -> TrajectoryInterpolator:
    """Load trajectory from TUM format file.

    Format: timestamp tx ty tz qx qy qz qw
    Lines starting with # are comments.

    Args:
        path: Path to TUM format file

    Returns:
        TrajectoryInterpolator instance
    """
    timestamps = []
    positions = []
    quaternions = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            values = [float(x) for x in line.split()]
            if len(values) != 8:
                continue

            t, tx, ty, tz, qx, qy, qz, qw = values
            timestamps.append(t)
            positions.append([tx, ty, tz])
            quaternions.append([qx, qy, qz, qw])

    return TrajectoryInterpolator(
        np.array(timestamps),
        np.array(positions),
        np.array(quaternions),
    )


def load_kitti_trajectory(path: Path, fps: float = 10.0) -> TrajectoryInterpolator:
    """Load trajectory from KITTI format file.

    Format: 12 values per line representing a 3x4 transformation matrix (row-major).
    Timestamps are assigned based on fps parameter.

    Args:
        path: Path to KITTI format file
        fps: Frame rate to assign timestamps (default 10 Hz for KITTI)

    Returns:
        TrajectoryInterpolator instance
    """
    timestamps = []
    positions = []
    quaternions = []

    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            values = [float(x) for x in line.split()]
            if len(values) != 12:
                raise ValueError(f"KITTI format expects 12 values per line, got {len(values)}")

            # Parse 3x4 matrix
            mat = np.array(values).reshape(3, 4)

            # Extract position and rotation
            position = mat[:, 3]
            rotation = Rotation.from_matrix(mat[:, :3])
            quaternion = rotation.as_quat()  # [qx, qy, qz, qw]

            timestamps.append(i / fps)
            positions.append(position)
            quaternions.append(quaternion)

    return TrajectoryInterpolator(
        np.array(timestamps),
        np.array(positions),
        np.array(quaternions),
    )


def load_vo_trajectory(path: Path) -> TrajectoryInterpolator:
    """Load trajectory from C++ VO output (trajectory.json).

    Format: JSON with "poses" array, each containing "timestamp", "translation", and "rotation".
    Each pose must include a timestamp field.

    Args:
        path: Path to trajectory.json file

    Returns:
        TrajectoryInterpolator instance

    Raises:
        KeyError: If a pose is missing the required "timestamp" field
    """
    with open(path) as f:
        data = json.load(f)

    # Handle both {"poses": [...]} and [...] formats
    if isinstance(data, dict) and "poses" in data:
        poses = data["poses"]
    else:
        poses = data

    timestamps = []
    positions = []
    quaternions = []

    for i, pose in enumerate(poses):
        if "timestamp" not in pose:
            raise KeyError(f"Pose {i} missing required 'timestamp' field")

        timestamp = pose["timestamp"]
        position = np.array(pose["translation"])
        rotation_mat = np.array(pose["rotation"]).reshape(3, 3)
        rotation = Rotation.from_matrix(rotation_mat)
        quaternion = rotation.as_quat()

        timestamps.append(timestamp)
        positions.append(position)
        quaternions.append(quaternion)

    return TrajectoryInterpolator(
        np.array(timestamps),
        np.array(positions),
        np.array(quaternions),
    )


def detect_format(path: Path) -> str:
    """Auto-detect trajectory file format.

    Args:
        path: Path to trajectory file

    Returns:
        Format string: 'tum', 'kitti', or 'vo_json'
    """
    if path.suffix == ".json":
        return "vo_json"

    with open(path) as f:
        first_line = f.readline().strip()

        # TUM files start with # comment or have 8 values
        if first_line.startswith("#"):
            return "tum"

        values = first_line.split()
        if len(values) == 8:
            return "tum"
        elif len(values) == 12:
            return "kitti"
        else:
            raise ValueError(
                f"Unknown format: {len(values)} values per line "
                "(expected 8 for TUM or 12 for KITTI)"
            )


def load_trajectory(
    path: Path,
    format: str = "auto",
    fps: float = 10.0,
) -> TrajectoryInterpolator:
    """Load trajectory from file with auto-detection.

    Args:
        path: Path to trajectory file
        format: Format hint ('auto', 'tum', 'kitti', 'vo_json')
        fps: Frame rate for formats without timestamps (KITTI only)

    Returns:
        TrajectoryInterpolator instance
    """
    path = Path(path)

    if format == "auto":
        format = detect_format(path)

    if format == "tum":
        return load_tum_trajectory(path)
    elif format == "kitti":
        return load_kitti_trajectory(path, fps)
    elif format == "vo_json":
        return load_vo_trajectory(path)
    else:
        raise ValueError(f"Unknown format: {format}")


@dataclass
class TrajectoryErrorResult:
    """Results from trajectory error computation."""

    timestamps: np.ndarray  # (N,) query timestamps
    position_errors: np.ndarray  # (N, 3) per-axis errors [dx, dy, dz]
    total_errors: np.ndarray  # (N,) euclidean distance errors
    ate_rmse: float  # Root mean square error
    ate_mean: float  # Mean error
    ate_median: float  # Median error
    ate_std: float  # Standard deviation
    ate_min: float  # Minimum error
    ate_max: float  # Maximum error


def compute_trajectory_error(
    ground_truth: TrajectoryInterpolator,
    estimated: TrajectoryInterpolator,
    query_timestamps: Optional[np.ndarray] = None,
) -> TrajectoryErrorResult:
    """Compute time-matched trajectory error between ground truth and estimated.

    Unlike sample-matched error, this properly interpolates poses at matching
    timestamps rather than assuming 1:1 frame correspondence.

    Args:
        ground_truth: Ground truth trajectory
        estimated: Estimated trajectory
        query_timestamps: Timestamps to evaluate at. If None, uses estimated timestamps.

    Returns:
        TrajectoryErrorResult with per-frame and aggregate metrics
    """
    if query_timestamps is None:
        query_timestamps = estimated.timestamps

    # Interpolate both trajectories at query times
    gt_positions, _ = ground_truth.interpolate_batch(query_timestamps)
    est_positions, _ = estimated.interpolate_batch(query_timestamps)

    # Compute per-axis errors
    position_errors = est_positions - gt_positions

    # Compute total (Euclidean) errors
    total_errors = np.linalg.norm(position_errors, axis=1)

    return TrajectoryErrorResult(
        timestamps=query_timestamps,
        position_errors=position_errors,
        total_errors=total_errors,
        ate_rmse=float(np.sqrt(np.mean(total_errors**2))),
        ate_mean=float(np.mean(total_errors)),
        ate_median=float(np.median(total_errors)),
        ate_std=float(np.std(total_errors)),
        ate_min=float(np.min(total_errors)),
        ate_max=float(np.max(total_errors)),
    )
