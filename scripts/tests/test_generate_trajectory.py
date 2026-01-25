#!/usr/bin/env python3
"""Unit tests for generate_test_trajectory.py"""

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_test_trajectory import (
    direction_to_quaternion,
    generate_line,
    generate_circle,
    generate_rectangle,
    generate_sinewave,
)
from scipy.spatial.transform import Rotation


class TestDirectionToQuaternion:
    """Tests for direction_to_quaternion function."""

    def test_forward_direction_identity(self):
        """Forward direction (+X) should give identity-ish quaternion."""
        # GIVEN forward direction
        direction = np.array([1.0, 0.0, 0.0])

        # WHEN converting to quaternion
        quat = direction_to_quaternion(direction)

        # THEN should be identity (or equivalent)
        rot = Rotation.from_quat(quat)
        # Apply rotation to +X, should still be +X
        result = rot.apply([1, 0, 0])
        np.testing.assert_array_almost_equal(result, [1, 0, 0], decimal=5)

    def test_backward_direction(self):
        """Backward direction (-X) should give 180 degree rotation."""
        # GIVEN backward direction
        direction = np.array([-1.0, 0.0, 0.0])

        # WHEN converting to quaternion
        quat = direction_to_quaternion(direction)

        # THEN rotation applied to +X should give -X
        rot = Rotation.from_quat(quat)
        result = rot.apply([1, 0, 0])
        np.testing.assert_array_almost_equal(result, [-1, 0, 0], decimal=5)

    def test_perpendicular_direction(self):
        """Perpendicular direction (+Y) should rotate 90 degrees."""
        # GIVEN +Y direction
        direction = np.array([0.0, 1.0, 0.0])

        # WHEN converting to quaternion
        quat = direction_to_quaternion(direction)

        # THEN rotation applied to +X should give +Y
        rot = Rotation.from_quat(quat)
        result = rot.apply([1, 0, 0])
        np.testing.assert_array_almost_equal(result, [0, 1, 0], decimal=5)

    def test_diagonal_direction(self):
        """Diagonal direction should point correctly."""
        # GIVEN diagonal direction
        direction = np.array([1.0, 1.0, 0.0])
        direction = direction / np.linalg.norm(direction)

        # WHEN converting to quaternion
        quat = direction_to_quaternion(direction)

        # THEN rotation applied to +X should give normalized diagonal
        rot = Rotation.from_quat(quat)
        result = rot.apply([1, 0, 0])
        np.testing.assert_array_almost_equal(result, direction, decimal=5)


class TestGenerateLine:
    """Tests for generate_line function."""

    def test_returns_correct_number_of_poses(self):
        """Should return requested number of poses."""
        # GIVEN/WHEN
        timestamps, positions, quaternions = generate_line(50, 1.0)

        # THEN
        assert len(timestamps) == 50
        assert len(positions) == 50
        assert len(quaternions) == 50

    def test_x_increases_linearly(self):
        """X coordinate should increase linearly."""
        # GIVEN/WHEN
        _, positions, _ = generate_line(100, 1.0)

        # THEN X should be monotonically increasing
        x = positions[:, 0]
        assert np.all(np.diff(x) > 0), "X should be strictly increasing"

        # AND should be approximately linear
        expected = np.linspace(x[0], x[-1], len(x))
        np.testing.assert_array_almost_equal(x, expected, decimal=5)

    def test_y_and_z_are_zero(self):
        """Y and Z should be zero for a line along X."""
        # GIVEN/WHEN
        _, positions, _ = generate_line(100, 1.0)

        # THEN
        np.testing.assert_array_almost_equal(positions[:, 1], 0, decimal=10)
        np.testing.assert_array_almost_equal(positions[:, 2], 0, decimal=10)

    def test_scale_affects_length(self):
        """Scale parameter should affect trajectory length."""
        # GIVEN two lines with different scales
        _, pos1, _ = generate_line(100, 1.0)
        _, pos2, _ = generate_line(100, 2.0)

        # THEN scaled line should be twice as long
        length1 = pos1[-1, 0] - pos1[0, 0]
        length2 = pos2[-1, 0] - pos2[0, 0]
        assert abs(length2 / length1 - 2.0) < 0.01


class TestGenerateCircle:
    """Tests for generate_circle function."""

    def test_returns_correct_number_of_poses(self):
        """Should return requested number of poses."""
        # GIVEN/WHEN
        timestamps, positions, quaternions = generate_circle(50, 1.0)

        # THEN
        assert len(timestamps) == 50
        assert len(positions) == 50
        assert len(quaternions) == 50

    def test_points_lie_on_circle(self):
        """All points should be equidistant from center."""
        # GIVEN/WHEN
        _, positions, _ = generate_circle(100, 1.0)

        # THEN all points should have same distance from origin (in XY plane)
        radii = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        expected_radius = radii[0]

        np.testing.assert_array_almost_equal(radii, expected_radius, decimal=5)

    def test_z_is_zero(self):
        """Z should be zero for circle in XY plane."""
        # GIVEN/WHEN
        _, positions, _ = generate_circle(100, 1.0)

        # THEN
        np.testing.assert_array_almost_equal(positions[:, 2], 0, decimal=10)

    def test_scale_affects_radius(self):
        """Scale parameter should affect radius."""
        # GIVEN two circles with different scales
        _, pos1, _ = generate_circle(100, 1.0)
        _, pos2, _ = generate_circle(100, 2.0)

        # THEN scaled circle should have twice the radius
        radius1 = np.sqrt(pos1[0, 0]**2 + pos1[0, 1]**2)
        radius2 = np.sqrt(pos2[0, 0]**2 + pos2[0, 1]**2)
        assert abs(radius2 / radius1 - 2.0) < 0.01

    def test_covers_full_circle(self):
        """Points should cover approximately 360 degrees."""
        # GIVEN/WHEN
        _, positions, _ = generate_circle(100, 1.0)

        # THEN angles should span nearly 2*pi
        angles = np.arctan2(positions[:, 1], positions[:, 0])
        # Unwrap to handle discontinuity at +/- pi
        angles_unwrapped = np.unwrap(angles)
        angle_span = abs(angles_unwrapped[-1] - angles_unwrapped[0])

        # Should be close to 2*pi (full circle minus one step)
        assert angle_span > 0.9 * 2 * np.pi


class TestGenerateRectangle:
    """Tests for generate_rectangle function."""

    def test_returns_correct_number_of_poses(self):
        """Should return requested number of poses."""
        # GIVEN/WHEN
        timestamps, positions, quaternions = generate_rectangle(100, 1.0)

        # THEN
        assert len(timestamps) == 100
        assert len(positions) == 100
        assert len(quaternions) == 100

    def test_z_is_zero(self):
        """Z should be zero for rectangle in XY plane."""
        # GIVEN/WHEN
        _, positions, _ = generate_rectangle(100, 1.0)

        # THEN
        np.testing.assert_array_almost_equal(positions[:, 2], 0, decimal=10)

    def test_stays_within_bounds(self):
        """Points should stay within rectangular bounds."""
        # GIVEN/WHEN
        _, positions, _ = generate_rectangle(100, 1.0)

        # THEN all points should be within expected bounds
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

        # Should form a rectangle (not a line or point)
        assert x_max - x_min > 0.1
        assert y_max - y_min > 0.1


class TestGenerateSinewave:
    """Tests for generate_sinewave function."""

    def test_returns_correct_number_of_poses(self):
        """Should return requested number of poses."""
        # GIVEN/WHEN
        timestamps, positions, quaternions = generate_sinewave(50, 1.0)

        # THEN
        assert len(timestamps) == 50
        assert len(positions) == 50
        assert len(quaternions) == 50

    def test_x_increases_monotonically(self):
        """X should increase monotonically (forward motion)."""
        # GIVEN/WHEN
        _, positions, _ = generate_sinewave(100, 1.0)

        # THEN X should be strictly increasing
        x = positions[:, 0]
        assert np.all(np.diff(x) > 0), "X should be strictly increasing"

    def test_y_oscillates(self):
        """Y should oscillate (sine pattern)."""
        # GIVEN/WHEN
        _, positions, _ = generate_sinewave(200, 1.0)

        # THEN Y should have both positive and negative values
        y = positions[:, 1]
        assert y.max() > 0
        assert y.min() < 0

        # AND should cross zero multiple times (oscillation)
        zero_crossings = np.sum(np.diff(np.sign(y)) != 0)
        assert zero_crossings >= 2, "Y should oscillate (cross zero multiple times)"

    def test_z_is_zero(self):
        """Z should be zero for sinewave in XY plane."""
        # GIVEN/WHEN
        _, positions, _ = generate_sinewave(100, 1.0)

        # THEN
        np.testing.assert_array_almost_equal(positions[:, 2], 0, decimal=10)


if __name__ == "__main__":
    print("Running tests for generate_test_trajectory.py...")

    print("\nTestDirectionToQuaternion...")
    t = TestDirectionToQuaternion()
    t.test_forward_direction_identity()
    print("  forward_direction_identity: PASS")
    t.test_backward_direction()
    print("  backward_direction: PASS")
    t.test_perpendicular_direction()
    print("  perpendicular_direction: PASS")
    t.test_diagonal_direction()
    print("  diagonal_direction: PASS")

    print("\nTestGenerateLine...")
    t = TestGenerateLine()
    t.test_returns_correct_number_of_poses()
    print("  returns_correct_number: PASS")
    t.test_x_increases_linearly()
    print("  x_increases_linearly: PASS")
    t.test_y_and_z_are_zero()
    print("  y_and_z_are_zero: PASS")
    t.test_scale_affects_length()
    print("  scale_affects_length: PASS")

    print("\nTestGenerateCircle...")
    t = TestGenerateCircle()
    t.test_returns_correct_number_of_poses()
    print("  returns_correct_number: PASS")
    t.test_points_lie_on_circle()
    print("  points_lie_on_circle: PASS")
    t.test_z_is_zero()
    print("  z_is_zero: PASS")
    t.test_scale_affects_radius()
    print("  scale_affects_radius: PASS")
    t.test_covers_full_circle()
    print("  covers_full_circle: PASS")

    print("\nTestGenerateRectangle...")
    t = TestGenerateRectangle()
    t.test_returns_correct_number_of_poses()
    print("  returns_correct_number: PASS")
    t.test_z_is_zero()
    print("  z_is_zero: PASS")
    t.test_stays_within_bounds()
    print("  stays_within_bounds: PASS")

    print("\nTestGenerateSinewave...")
    t = TestGenerateSinewave()
    t.test_returns_correct_number_of_poses()
    print("  returns_correct_number: PASS")
    t.test_x_increases_monotonically()
    print("  x_increases_monotonically: PASS")
    t.test_y_oscillates()
    print("  y_oscillates: PASS")
    t.test_z_is_zero()
    print("  z_is_zero: PASS")

    print("\n" + "="*50)
    print("All tests PASSED!")
