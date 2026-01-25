#!/usr/bin/env python3
"""Unit tests for plot_trajectory.py timestamp matching functions."""

import numpy as np
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from plot_trajectory import (
    get_image_paths_with_timestamps,
    match_images_to_timestamps,
)


class TestGetImagePathsWithTimestamps:
    """Tests for get_image_paths_with_timestamps function."""

    def test_parses_tum_style_timestamps(self):
        """Should parse TUM-style timestamp filenames."""
        # GIVEN a directory with TUM-style named images
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "1305031102.175304.png").touch()
            (tmpdir / "1305031102.211214.png").touch()
            (tmpdir / "1305031102.243211.png").touch()

            # WHEN getting image paths
            result = get_image_paths_with_timestamps(tmpdir)

            # THEN timestamps should be parsed correctly
            assert len(result) == 3
            assert abs(result[0][0] - 1305031102.175304) < 0.000001
            assert abs(result[1][0] - 1305031102.211214) < 0.000001
            assert abs(result[2][0] - 1305031102.243211) < 0.000001

    def test_falls_back_to_index_for_non_timestamp_names(self):
        """Should use index as timestamp for non-parseable filenames."""
        # GIVEN a directory with regular named images
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "000000.png").touch()
            (tmpdir / "000001.png").touch()
            (tmpdir / "000002.png").touch()

            # WHEN getting image paths
            result = get_image_paths_with_timestamps(tmpdir)

            # THEN timestamps should be indices
            assert len(result) == 3
            timestamps = [t for t, _ in result]
            assert timestamps == [0.0, 1.0, 2.0]

    def test_returns_sorted_by_timestamp(self):
        """Should return list sorted by timestamp."""
        # GIVEN images with out-of-order timestamps
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "1305031105.000000.png").touch()
            (tmpdir / "1305031102.000000.png").touch()
            (tmpdir / "1305031108.000000.png").touch()

            # WHEN getting image paths
            result = get_image_paths_with_timestamps(tmpdir)

            # THEN should be sorted by timestamp
            timestamps = [t for t, _ in result]
            assert timestamps == sorted(timestamps)

    def test_handles_empty_directory(self):
        """Should handle empty directory."""
        # GIVEN an empty directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # WHEN getting image paths
            result = get_image_paths_with_timestamps(tmpdir)

            # THEN should return empty list
            assert result == []

    def test_filters_to_image_extensions(self):
        """Should only include image files."""
        # GIVEN a directory with mixed files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "1305031102.175304.png").touch()
            (tmpdir / "1305031102.211214.jpg").touch()
            (tmpdir / "1305031102.243211.txt").touch()  # Not an image
            (tmpdir / "readme.md").touch()  # Not an image

            # WHEN getting image paths
            result = get_image_paths_with_timestamps(tmpdir)

            # THEN should only include image files
            assert len(result) == 2


class TestMatchImagesToTimestamps:
    """Tests for match_images_to_timestamps function."""

    def test_exact_timestamp_match(self):
        """Should return correct image for exact timestamp match."""
        # GIVEN trajectory timestamps matching image timestamps exactly
        trajectory_timestamps = [1.0, 2.0, 3.0]
        image_data = [
            (1.0, "/path/to/img1.png"),
            (2.0, "/path/to/img2.png"),
            (3.0, "/path/to/img3.png"),
        ]

        # WHEN matching
        result = match_images_to_timestamps(trajectory_timestamps, image_data)

        # THEN should return exact matches
        assert result == ["/path/to/img1.png", "/path/to/img2.png", "/path/to/img3.png"]

    def test_closest_image_selected(self):
        """Should select closest image when no exact match."""
        # GIVEN trajectory timestamps between image timestamps
        trajectory_timestamps = [1.5, 2.5]  # Between images
        image_data = [
            (1.0, "/path/to/img1.png"),
            (2.0, "/path/to/img2.png"),
            (3.0, "/path/to/img3.png"),
        ]

        # WHEN matching with large max_time_diff
        result = match_images_to_timestamps(trajectory_timestamps, image_data, max_time_diff=1.0)

        # THEN should select closest images
        assert result[0] == "/path/to/img1.png" or result[0] == "/path/to/img2.png"  # 1.5 is equidistant
        assert result[1] == "/path/to/img2.png" or result[1] == "/path/to/img3.png"  # 2.5 is equidistant

    def test_returns_none_outside_range(self):
        """Should return None when timestamp is outside max_time_diff."""
        # GIVEN trajectory timestamps far from image timestamps
        trajectory_timestamps = [0.0, 5.0, 10.0]
        image_data = [
            (2.0, "/path/to/img1.png"),
            (3.0, "/path/to/img2.png"),
        ]

        # WHEN matching with small max_time_diff
        result = match_images_to_timestamps(trajectory_timestamps, image_data, max_time_diff=0.5)

        # THEN timestamps outside range should return None
        assert result[0] is None  # 0.0 is far from 2.0
        assert result[1] is None  # 5.0 is far from 3.0
        assert result[2] is None  # 10.0 is far from 3.0

    def test_handles_empty_image_list(self):
        """Should return all None for empty image list."""
        # GIVEN trajectory timestamps but no images
        trajectory_timestamps = [1.0, 2.0, 3.0]
        image_data = []

        # WHEN matching
        result = match_images_to_timestamps(trajectory_timestamps, image_data)

        # THEN should return all None
        assert result == [None, None, None]

    def test_many_to_one_matching(self):
        """Multiple trajectory timestamps can match same image."""
        # GIVEN more trajectory timestamps than images
        trajectory_timestamps = [1.0, 1.01, 1.02, 1.03]  # All close to 1.0
        image_data = [
            (1.0, "/path/to/img1.png"),
            (2.0, "/path/to/img2.png"),
        ]

        # WHEN matching
        result = match_images_to_timestamps(trajectory_timestamps, image_data, max_time_diff=0.1)

        # THEN all should match first image
        assert result == ["/path/to/img1.png"] * 4

    def test_boundary_max_time_diff(self):
        """Should include matches within max_time_diff boundary."""
        # GIVEN timestamp within max_time_diff boundary
        trajectory_timestamps = [1.05]  # 0.05 diff from image
        image_data = [(1.0, "/path/to/img.png")]

        # WHEN matching with max_time_diff = 0.1
        result = match_images_to_timestamps(trajectory_timestamps, image_data, max_time_diff=0.1)

        # THEN should match (0.05 <= 0.1)
        assert result == ["/path/to/img.png"]

    def test_just_outside_max_time_diff(self):
        """Should not match when just outside max_time_diff."""
        # GIVEN timestamp just outside max_time_diff
        trajectory_timestamps = [1.2]  # 0.2 diff from image
        image_data = [(1.0, "/path/to/img.png")]

        # WHEN matching with max_time_diff = 0.1
        result = match_images_to_timestamps(trajectory_timestamps, image_data, max_time_diff=0.1)

        # THEN should not match
        assert result == [None]


if __name__ == "__main__":
    print("Running tests for plot_trajectory.py...")

    print("\nTestGetImagePathsWithTimestamps...")
    t = TestGetImagePathsWithTimestamps()
    t.test_parses_tum_style_timestamps()
    print("  parses_tum_style_timestamps: PASS")
    t.test_falls_back_to_index_for_non_timestamp_names()
    print("  falls_back_to_index: PASS")
    t.test_returns_sorted_by_timestamp()
    print("  returns_sorted: PASS")
    t.test_handles_empty_directory()
    print("  handles_empty_directory: PASS")
    t.test_filters_to_image_extensions()
    print("  filters_to_image_extensions: PASS")

    print("\nTestMatchImagesToTimestamps...")
    t = TestMatchImagesToTimestamps()
    t.test_exact_timestamp_match()
    print("  exact_timestamp_match: PASS")
    t.test_closest_image_selected()
    print("  closest_image_selected: PASS")
    t.test_returns_none_outside_range()
    print("  returns_none_outside_range: PASS")
    t.test_handles_empty_image_list()
    print("  handles_empty_image_list: PASS")
    t.test_many_to_one_matching()
    print("  many_to_one_matching: PASS")
    t.test_boundary_max_time_diff()
    print("  boundary_max_time_diff: PASS")
    t.test_just_outside_max_time_diff()
    print("  just_outside_max_time_diff: PASS")

    print("\n" + "="*50)
    print("All tests PASSED!")
