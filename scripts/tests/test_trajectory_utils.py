#!/usr/bin/env python3
"""Unit tests for trajectory_utils module."""

import numpy as np
import tempfile

try:
    import pytest
except ImportError:
    pytest = None
from pathlib import Path
from scipy.spatial.transform import Rotation

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from trajectory_utils import (
    TrajectoryInterpolator,
    Pose,
    load_tum_trajectory,
    load_kitti_trajectory,
    load_vo_trajectory,
    load_trajectory,
    compute_trajectory_error,
)


class TestTrajectoryInterpolator:
    """Tests for TrajectoryInterpolator class."""

    def test_single_pose_returns_that_pose(self):
        """Single-pose trajectory always returns that pose."""
        # GIVEN a single-pose trajectory
        timestamps = np.array([1.0])
        positions = np.array([[1.0, 2.0, 3.0]])
        quaternions = np.array([[0.0, 0.0, 0.0, 1.0]])

        interp = TrajectoryInterpolator(timestamps, positions, quaternions)

        # WHEN interpolating at any time
        pose = interp.interpolate(0.0)

        # THEN returns the single pose
        np.testing.assert_array_almost_equal(pose.position, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(pose.quaternion, [0.0, 0.0, 0.0, 1.0])

    def test_exact_timestamp_returns_exact_pose(self):
        """Interpolating at exact timestamp returns exact pose."""
        # GIVEN a multi-pose trajectory
        timestamps = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        quaternions = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        interp = TrajectoryInterpolator(timestamps, positions, quaternions)

        # WHEN interpolating at exact timestamp
        pose = interp.interpolate(1.0)

        # THEN returns exact pose
        np.testing.assert_array_almost_equal(pose.position, [1.0, 0.0, 0.0])

    def test_linear_interpolation_position(self):
        """Position is linearly interpolated between timestamps."""
        # GIVEN a trajectory with two poses
        timestamps = np.array([0.0, 1.0])
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
        quaternions = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        interp = TrajectoryInterpolator(timestamps, positions, quaternions)

        # WHEN interpolating at midpoint
        pose = interp.interpolate(0.5)

        # THEN position is halfway
        np.testing.assert_array_almost_equal(pose.position, [1.0, 2.0, 3.0])

    def test_slerp_interpolation_quaternion(self):
        """Quaternion is slerp interpolated between timestamps."""
        # GIVEN a trajectory with 90-degree rotation
        timestamps = np.array([0.0, 1.0])
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        # Identity to 90-degree rotation around Z
        q1 = [0.0, 0.0, 0.0, 1.0]  # identity
        q2 = Rotation.from_euler("z", 90, degrees=True).as_quat()
        quaternions = np.array([q1, q2])

        interp = TrajectoryInterpolator(timestamps, positions, quaternions)

        # WHEN interpolating at midpoint
        pose = interp.interpolate(0.5)

        # THEN quaternion is 45-degree rotation
        expected_rot = Rotation.from_euler("z", 45, degrees=True)
        actual_rot = Rotation.from_quat(pose.quaternion)

        # Compare rotation matrices (handles quaternion sign ambiguity)
        np.testing.assert_array_almost_equal(
            actual_rot.as_matrix(),
            expected_rot.as_matrix(),
            decimal=5,
        )

    def test_before_first_timestamp_returns_first_pose(self):
        """Query before trajectory start returns first pose."""
        # GIVEN a trajectory
        timestamps = np.array([1.0, 2.0])
        positions = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        quaternions = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        interp = TrajectoryInterpolator(timestamps, positions, quaternions)

        # WHEN interpolating before start
        pose = interp.interpolate(0.0)

        # THEN returns first pose
        np.testing.assert_array_almost_equal(pose.position, [1.0, 0.0, 0.0])

    def test_after_last_timestamp_returns_last_pose(self):
        """Query after trajectory end returns last pose."""
        # GIVEN a trajectory
        timestamps = np.array([1.0, 2.0])
        positions = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        quaternions = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        interp = TrajectoryInterpolator(timestamps, positions, quaternions)

        # WHEN interpolating after end
        pose = interp.interpolate(10.0)

        # THEN returns last pose
        np.testing.assert_array_almost_equal(pose.position, [2.0, 0.0, 0.0])

    def test_batch_interpolation(self):
        """Batch interpolation returns correct shape and values."""
        # GIVEN a trajectory
        timestamps = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        quaternions = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        interp = TrajectoryInterpolator(timestamps, positions, quaternions)

        # WHEN batch interpolating
        query_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        positions_out, quaternions_out = interp.interpolate_batch(query_times)

        # THEN shapes are correct
        assert positions_out.shape == (5, 3)
        assert quaternions_out.shape == (5, 4)

        # AND values are correct
        np.testing.assert_array_almost_equal(positions_out[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(positions_out[1], [0.5, 0.0, 0.0])
        np.testing.assert_array_almost_equal(positions_out[2], [1.0, 0.0, 0.0])

    def test_duration_property(self):
        """Duration property returns correct value."""
        # GIVEN a trajectory
        timestamps = np.array([1.0, 2.0, 5.0])
        positions = np.zeros((3, 3))
        quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))

        interp = TrajectoryInterpolator(timestamps, positions, quaternions)

        # THEN duration is correct
        assert interp.duration == 4.0
        assert interp.start_time == 1.0
        assert interp.end_time == 5.0


class TestTumLoader:
    """Tests for TUM format loader."""

    def test_load_tum_trajectory(self):
        """Load valid TUM format file."""
        # GIVEN a TUM format file
        content = """# comment line
0.0 1.0 2.0 3.0 0.0 0.0 0.0 1.0
1.0 4.0 5.0 6.0 0.0 0.0 0.0 1.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = Path(f.name)

        try:
            # WHEN loading
            interp = load_tum_trajectory(path)

            # THEN trajectory is correct
            assert len(interp) == 2
            np.testing.assert_array_almost_equal(interp.timestamps, [0.0, 1.0])
            np.testing.assert_array_almost_equal(interp.positions[0], [1.0, 2.0, 3.0])
        finally:
            path.unlink()


class TestKittiLoader:
    """Tests for KITTI format loader."""

    def test_load_kitti_trajectory(self):
        """Load valid KITTI format file."""
        # GIVEN a KITTI format file (identity poses)
        content = """1 0 0 1 0 1 0 2 0 0 1 3
1 0 0 4 0 1 0 5 0 0 1 6
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = Path(f.name)

        try:
            # WHEN loading with 10 fps
            interp = load_kitti_trajectory(path, fps=10.0)

            # THEN trajectory is correct
            assert len(interp) == 2
            np.testing.assert_array_almost_equal(interp.timestamps, [0.0, 0.1])
            np.testing.assert_array_almost_equal(interp.positions[0], [1.0, 2.0, 3.0])
            np.testing.assert_array_almost_equal(interp.positions[1], [4.0, 5.0, 6.0])
        finally:
            path.unlink()


class TestTrajectoryError:
    """Tests for trajectory error computation."""

    def test_identical_trajectories_zero_error(self):
        """Identical trajectories have zero error."""
        # GIVEN two identical trajectories
        timestamps = np.array([0.0, 1.0, 2.0])
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))

        gt = TrajectoryInterpolator(timestamps, positions, quaternions)
        est = TrajectoryInterpolator(timestamps, positions, quaternions)

        # WHEN computing error
        result = compute_trajectory_error(gt, est)

        # THEN error is zero
        assert abs(result.ate_rmse) < 1e-10
        assert abs(result.ate_mean) < 1e-10
        np.testing.assert_array_almost_equal(result.total_errors, [0.0, 0.0, 0.0])

    def test_known_offset_error(self):
        """Known constant offset produces expected error."""
        # GIVEN ground truth at origin
        timestamps = np.array([0.0, 1.0])
        gt_positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))

        gt = TrajectoryInterpolator(timestamps, gt_positions, quaternions)

        # AND estimated with 3-4-5 triangle offset (error = 5)
        est_positions = np.array([[3.0, 4.0, 0.0], [3.0, 4.0, 0.0]])
        est = TrajectoryInterpolator(timestamps, est_positions, quaternions)

        # WHEN computing error
        result = compute_trajectory_error(gt, est)

        # THEN error is 5.0
        assert abs(result.ate_rmse - 5.0) < 1e-10
        assert abs(result.ate_mean - 5.0) < 1e-10

    def test_different_sample_rates(self):
        """Different sample rates handled via interpolation."""
        # GIVEN ground truth with many samples
        gt_times = np.linspace(0.0, 1.0, 100)
        gt_positions = np.column_stack([gt_times, np.zeros((100, 2))])  # X = t
        gt_quats = np.tile([0.0, 0.0, 0.0, 1.0], (100, 1))
        gt = TrajectoryInterpolator(gt_times, gt_positions, gt_quats)

        # AND estimated with few samples (but same trajectory)
        est_times = np.array([0.0, 0.5, 1.0])
        est_positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
        est_quats = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))
        est = TrajectoryInterpolator(est_times, est_positions, est_quats)

        # WHEN computing error at ground truth timestamps
        result = compute_trajectory_error(gt, est, query_timestamps=gt_times)

        # THEN error is near zero (interpolation matches)
        assert abs(result.ate_rmse) < 1e-6

    def test_per_axis_errors(self):
        """Per-axis errors are computed correctly."""
        # GIVEN trajectories with known per-axis offsets
        timestamps = np.array([0.0])
        gt_positions = np.array([[0.0, 0.0, 0.0]])
        est_positions = np.array([[1.0, 2.0, 3.0]])
        quaternions = np.array([[0.0, 0.0, 0.0, 1.0]])

        gt = TrajectoryInterpolator(timestamps, gt_positions, quaternions)
        est = TrajectoryInterpolator(timestamps, est_positions, quaternions)

        # WHEN computing error
        result = compute_trajectory_error(gt, est)

        # THEN per-axis errors are correct
        np.testing.assert_array_almost_equal(result.position_errors[0], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    if pytest:
        pytest.main([__file__, "-v"])
    else:
        # Run tests manually without pytest
        print("Running tests without pytest...")

        print("Testing TrajectoryInterpolator...")
        t = TestTrajectoryInterpolator()
        t.test_single_pose_returns_that_pose()
        print("  single_pose: PASS")
        t.test_exact_timestamp_returns_exact_pose()
        print("  exact_timestamp: PASS")
        t.test_linear_interpolation_position()
        print("  linear_interp: PASS")
        t.test_slerp_interpolation_quaternion()
        print("  slerp: PASS")
        t.test_before_first_timestamp_returns_first_pose()
        print("  before_first: PASS")
        t.test_after_last_timestamp_returns_last_pose()
        print("  after_last: PASS")
        t.test_batch_interpolation()
        print("  batch: PASS")
        t.test_duration_property()
        print("  duration: PASS")

        print("Testing TumLoader...")
        t = TestTumLoader()
        t.test_load_tum_trajectory()
        print("  load_tum: PASS")

        print("Testing KittiLoader...")
        t = TestKittiLoader()
        t.test_load_kitti_trajectory()
        print("  load_kitti: PASS")

        print("Testing TrajectoryError...")
        t = TestTrajectoryError()
        t.test_identical_trajectories_zero_error()
        print("  identical_zero: PASS")
        t.test_known_offset_error()
        print("  known_offset: PASS")
        t.test_different_sample_rates()
        print("  different_rates: PASS")
        t.test_per_axis_errors()
        print("  per_axis: PASS")

        print()
        print("All tests PASSED!")
