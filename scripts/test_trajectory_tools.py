#!/usr/bin/env python3
"""Integration tests for trajectory comparison tools.

Tests both plot_trajectory_error.py and compare_trajectories.py to verify:
- HTML generation and controls
- Error metric computation
- Umeyama alignment functions
- Module import without GUI activation
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from trajectory_utils import (
    TrajectoryInterpolator,
    TrajectoryErrorResult,
    compute_trajectory_error,
)

# Import functions from trajectory tools (without starting servers)
from plot_trajectory_error import (
    compute_umeyama_alignment as umeyama_html,
    create_plot_html,
)

# Import compare_trajectories functions conditionally (viser might not be available)
try:
    from compare_trajectories import (
        compute_umeyama_alignment as umeyama_viser,
        compute_scaled_error,
    )
    VISER_AVAILABLE = True
except ImportError:
    VISER_AVAILABLE = False
    umeyama_viser = None
    compute_scaled_error = None


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_trajectory():
    """Create a simple linear trajectory for testing."""
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ])
    quaternions = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return TrajectoryInterpolator(timestamps, positions, quaternions)


@pytest.fixture
def scaled_trajectory():
    """Create a scaled version of the simple trajectory (scale=2.0)."""
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
    ])
    quaternions = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return TrajectoryInterpolator(timestamps, positions, quaternions)


@pytest.fixture
def offset_trajectory():
    """Create an offset version of the simple trajectory (+1.0 in Y)."""
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    positions = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [3.0, 1.0, 0.0],
        [4.0, 1.0, 0.0],
    ])
    quaternions = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return TrajectoryInterpolator(timestamps, positions, quaternions)


@pytest.fixture
def temp_trajectory_file():
    """Create a temporary trajectory JSON file with known data."""
    data = {
        "poses": [
            {
                "timestamp": 0.0,
                "translation": [0.0, 0.0, 0.0],
                "rotation": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0
                ]
            },
            {
                "timestamp": 1.0,
                "translation": [1.0, 0.0, 0.0],
                "rotation": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0
                ]
            },
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    temp_path.unlink(missing_ok=True)


@pytest.fixture
def temp_trajectory_file_no_timestamp():
    """Create a temporary trajectory JSON file missing timestamp field."""
    data = {
        "poses": [
            {
                "translation": [0.0, 0.0, 0.0],
                "rotation": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0
                ]
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    temp_path.unlink(missing_ok=True)


# ============================================================================
# Test plot_trajectory_error.py
# ============================================================================

class TestPlotTrajectoryError:
    """Tests for plot_trajectory_error.py HTML generation and error computation."""

    def test_html_generation_produces_valid_output(self, simple_trajectory):
        """Test that create_plot_html generates valid HTML string."""
        # GIVEN two identical trajectories
        gt_interp = simple_trajectory
        est_interp = simple_trajectory
        error_result = compute_trajectory_error(gt_interp, est_interp)
        
        # WHEN generating HTML
        html = create_plot_html(gt_interp, est_interp, error_result, scale=1.0)
        
        # THEN it should be a non-empty string
        assert isinstance(html, str)
        assert len(html) > 0
        
        # AND it should contain basic HTML structure
        assert '<!DOCTYPE html>' in html
        assert '<html>' in html
        assert '</html>' in html
        assert '<head>' in html
        assert '<body>' in html

    def test_html_contains_plotly_library(self, simple_trajectory):
        """Test that HTML includes plotly library for interactive plots."""
        # GIVEN simple trajectories
        gt_interp = simple_trajectory
        est_interp = simple_trajectory
        error_result = compute_trajectory_error(gt_interp, est_interp)
        
        # WHEN generating HTML
        html = create_plot_html(gt_interp, est_interp, error_result)
        
        # THEN it should include plotly CDN
        assert 'plotly' in html.lower()
        assert 'cdn.plot.ly' in html or 'plotly-2' in html

    def test_html_contains_scale_slider_control(self, simple_trajectory):
        """Test that HTML contains scale slider control."""
        # GIVEN simple trajectories
        gt_interp = simple_trajectory
        est_interp = simple_trajectory
        error_result = compute_trajectory_error(gt_interp, est_interp)
        
        # WHEN generating HTML
        html = create_plot_html(gt_interp, est_interp, error_result)
        
        # THEN it should contain scale slider elements
        assert 'scale-slider' in html or 'Estimated Scale' in html or 'Scale' in html
        assert 'type="range"' in html
        assert 'scale-input' in html or 'type="number"' in html

    def test_html_contains_offset_controls(self, simple_trajectory):
        """Test that HTML contains X/Y/Z offset sliders."""
        # GIVEN simple trajectories
        gt_interp = simple_trajectory
        est_interp = simple_trajectory
        error_result = compute_trajectory_error(gt_interp, est_interp)
        
        # WHEN generating HTML
        html = create_plot_html(gt_interp, est_interp, error_result)
        
        # THEN it should contain offset controls
        assert 'offset-x-slider' in html or 'Offset X' in html
        assert 'offset-y-slider' in html or 'Offset Y' in html
        assert 'offset-z-slider' in html or 'Offset Z' in html

    def test_html_contains_auto_align_button(self, simple_trajectory):
        """Test that HTML contains auto-align button for Umeyama alignment."""
        # GIVEN simple trajectories
        gt_interp = simple_trajectory
        est_interp = simple_trajectory
        error_result = compute_trajectory_error(gt_interp, est_interp)
        
        # WHEN generating HTML
        html = create_plot_html(gt_interp, est_interp, error_result)
        
        # THEN it should contain auto-align button
        assert 'auto-align' in html.lower() or 'umeyama' in html.lower()
        assert '<button' in html.lower()

    def test_html_displays_error_metrics(self, simple_trajectory):
        """Test that HTML displays error metrics (RMSE, mean, median, etc.)."""
        # GIVEN simple trajectories
        gt_interp = simple_trajectory
        est_interp = simple_trajectory
        error_result = compute_trajectory_error(gt_interp, est_interp)
        
        # WHEN generating HTML
        html = create_plot_html(gt_interp, est_interp, error_result)
        
        # THEN it should display error metrics
        assert 'RMSE' in html or 'rmse' in html.lower()
        assert 'Mean' in html or 'mean' in html.lower()
        assert 'Median' in html or 'median' in html.lower()
        assert str(round(error_result.ate_rmse, 4)) in html

    def test_error_metrics_computed_correctly_zero_error(self, simple_trajectory):
        """Test error computation with identical trajectories (zero error)."""
        # GIVEN two identical trajectories
        gt_interp = simple_trajectory
        est_interp = simple_trajectory
        
        # WHEN computing error
        error_result = compute_trajectory_error(gt_interp, est_interp)
        
        # THEN all errors should be zero (or very close to zero)
        assert error_result.ate_rmse < 1e-10
        assert error_result.ate_mean < 1e-10
        assert error_result.ate_median < 1e-10
        assert error_result.ate_min < 1e-10
        assert error_result.ate_max < 1e-10

    def test_error_metrics_with_offset_trajectory(self, simple_trajectory, offset_trajectory):
        """Test error computation with known offset (Y+1.0)."""
        # GIVEN ground truth and offset trajectory
        gt_interp = simple_trajectory
        est_interp = offset_trajectory
        
        # WHEN computing error
        error_result = compute_trajectory_error(gt_interp, est_interp)
        
        # THEN error should be 1.0 (constant Y offset)
        assert abs(error_result.ate_rmse - 1.0) < 1e-6
        assert abs(error_result.ate_mean - 1.0) < 1e-6
        assert abs(error_result.ate_median - 1.0) < 1e-6
        assert abs(error_result.ate_min - 1.0) < 1e-6
        assert abs(error_result.ate_max - 1.0) < 1e-6

    def test_umeyama_alignment_identity_case(self, simple_trajectory):
        """Test Umeyama alignment with identical trajectories."""
        # GIVEN two identical trajectories
        gt_positions = simple_trajectory.positions
        est_positions = simple_trajectory.positions
        
        # WHEN computing Umeyama alignment
        scale, rotation, translation = umeyama_html(gt_positions, est_positions)
        
        # THEN scale should be 1.0
        assert abs(scale - 1.0) < 1e-6
        
        # AND rotation should be identity
        expected_identity = np.eye(3)
        assert np.allclose(rotation, expected_identity, atol=1e-6)
        
        # AND translation should be near zero
        assert np.allclose(translation, np.zeros(3), atol=1e-6)

    def test_umeyama_alignment_with_scale(self, simple_trajectory, scaled_trajectory):
        """Test Umeyama alignment with scaled trajectory (scale=2.0)."""
        # GIVEN ground truth and scaled estimated trajectory
        gt_positions = scaled_trajectory.positions  # scale=2.0
        est_positions = simple_trajectory.positions  # scale=1.0
        
        # WHEN computing Umeyama alignment
        scale, rotation, translation = umeyama_html(gt_positions, est_positions)
        
        # THEN scale should be approximately 2.0
        assert abs(scale - 2.0) < 1e-6
        
        # AND rotation should be identity (no rotation)
        expected_identity = np.eye(3)
        assert np.allclose(rotation, expected_identity, atol=1e-6)

    def test_missing_timestamp_raises_error(self, temp_trajectory_file_no_timestamp):
        """Test that loading trajectory without timestamp raises KeyError."""
        # GIVEN a trajectory JSON file missing timestamp field
        from trajectory_utils import load_trajectory
        
        # WHEN loading the trajectory
        # THEN it should raise KeyError with clear message
        with pytest.raises(KeyError) as exc_info:
            load_trajectory(temp_trajectory_file_no_timestamp)
        
        assert 'timestamp' in str(exc_info.value).lower()


# ============================================================================
# Test compare_trajectories.py
# ============================================================================

@pytest.mark.skipif(not VISER_AVAILABLE, reason="viser not available")
class TestCompareTrajectories:
    """Tests for compare_trajectories.py viser visualization tool."""

    def test_module_imports_without_starting_server(self):
        """Test that compare_trajectories can be imported without starting viser server."""
        # WHEN importing the module
        import compare_trajectories
        
        # THEN it should import successfully without starting a server
        assert hasattr(compare_trajectories, 'compute_umeyama_alignment')
        assert hasattr(compare_trajectories, 'compute_scaled_error')
        assert hasattr(compare_trajectories, 'add_trajectory_to_scene')
        assert hasattr(compare_trajectories, 'remove_trajectory_from_scene')

    def test_compute_umeyama_alignment_function_exists(self):
        """Test that compute_umeyama_alignment function is accessible."""
        # THEN function should be callable
        assert callable(umeyama_viser)

    def test_compute_umeyama_alignment_identity(self, simple_trajectory):
        """Test Umeyama alignment with identical point sets."""
        # GIVEN two identical trajectories
        gt_positions = simple_trajectory.positions
        est_positions = simple_trajectory.positions
        
        # WHEN computing alignment
        scale, rotation, translation = umeyama_viser(gt_positions, est_positions)
        
        # THEN scale should be 1.0
        assert abs(scale - 1.0) < 1e-6
        
        # AND rotation should be identity
        assert np.allclose(rotation, np.eye(3), atol=1e-6)
        
        # AND translation should be near zero
        assert np.allclose(translation, np.zeros(3), atol=1e-6)

    def test_compute_umeyama_alignment_with_scale_factor(self, simple_trajectory, scaled_trajectory):
        """Test Umeyama alignment recovers correct scale factor."""
        # GIVEN ground truth (scale=2.0) and estimated (scale=1.0)
        gt_positions = scaled_trajectory.positions
        est_positions = simple_trajectory.positions
        
        # WHEN computing alignment
        scale, rotation, translation = umeyama_viser(gt_positions, est_positions)
        
        # THEN scale should be approximately 2.0
        assert abs(scale - 2.0) < 1e-6

    def test_compute_scaled_error_no_scaling(self, simple_trajectory):
        """Test compute_scaled_error with scale=1.0 and no offset."""
        # GIVEN two identical trajectories
        gt_interp = simple_trajectory
        est_interp = simple_trajectory
        
        # WHEN computing scaled error with scale=1.0, offset=0
        error_result = compute_scaled_error(gt_interp, est_interp, scale=1.0)
        
        # THEN error should be near zero
        assert error_result.ate_rmse < 1e-10
        assert error_result.ate_mean < 1e-10

    def test_compute_scaled_error_with_scale_factor(self, simple_trajectory, scaled_trajectory):
        """Test compute_scaled_error corrects for scale factor."""
        # GIVEN ground truth (scale=2.0) and estimated (scale=1.0)
        gt_interp = scaled_trajectory
        est_interp = simple_trajectory
        
        # WHEN applying correct scale factor (2.0)
        error_result = compute_scaled_error(gt_interp, est_interp, scale=2.0)
        
        # THEN error should be near zero
        assert error_result.ate_rmse < 1e-6
        assert error_result.ate_mean < 1e-6

    def test_compute_scaled_error_with_offset(self, simple_trajectory):
        """Test compute_scaled_error with translation offset."""
        # GIVEN ground truth and estimated trajectory
        gt_interp = simple_trajectory
        est_interp = simple_trajectory
        
        # WHEN applying Y offset of -1.0
        offset = np.array([0.0, -1.0, 0.0])
        error_result = compute_scaled_error(gt_interp, est_interp, scale=1.0, offset=offset)
        
        # THEN error should be 1.0 (constant offset)
        assert abs(error_result.ate_rmse - 1.0) < 1e-6

    @pytest.mark.skipif(not VISER_AVAILABLE, reason="viser not available")
    def test_umeyama_consistency_between_modules(self, simple_trajectory, scaled_trajectory):
        """Test that both umeyama implementations give same results."""
        # GIVEN ground truth and estimated positions
        gt_positions = scaled_trajectory.positions
        est_positions = simple_trajectory.positions
        
        # WHEN computing alignment with both implementations
        scale_html, rotation_html, translation_html = umeyama_html(gt_positions, est_positions)
        scale_viser, rotation_viser, translation_viser = umeyama_viser(gt_positions, est_positions)
        
        # THEN results should match
        assert abs(scale_html - scale_viser) < 1e-10
        assert np.allclose(rotation_html, rotation_viser, atol=1e-10)
        assert np.allclose(translation_html, translation_viser, atol=1e-10)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests that verify end-to-end workflows."""

    def test_load_trajectory_and_compute_error(self, temp_trajectory_file):
        """Test complete workflow: load trajectory and compute error."""
        # GIVEN a trajectory file
        from trajectory_utils import load_trajectory
        
        # WHEN loading trajectory
        interp = load_trajectory(temp_trajectory_file)
        
        # THEN it should have expected properties
        assert len(interp) == 2
        assert interp.duration > 0
        
        # AND we can compute error against itself (should be zero)
        error_result = compute_trajectory_error(interp, interp)
        assert error_result.ate_rmse < 1e-10

    def test_html_generation_with_real_error_data(self, simple_trajectory, offset_trajectory):
        """Test HTML generation with real error data."""
        # GIVEN ground truth and offset estimated trajectory
        gt_interp = simple_trajectory
        est_interp = offset_trajectory
        error_result = compute_trajectory_error(gt_interp, est_interp)
        
        # WHEN generating HTML
        html = create_plot_html(gt_interp, est_interp, error_result)
        
        # THEN HTML should contain the computed error values
        assert str(round(error_result.ate_rmse, 4)) in html
        
        # AND it should contain plot divs
        assert 'plot_x' in html
        assert 'plot_y' in html
        assert 'plot_z' in html
        assert 'plot_error' in html

    def test_deterministic_error_computation(self, simple_trajectory):
        """Test that error computation is deterministic."""
        # GIVEN a trajectory
        gt_interp = simple_trajectory
        est_interp = simple_trajectory
        
        # WHEN computing error multiple times
        result1 = compute_trajectory_error(gt_interp, est_interp)
        result2 = compute_trajectory_error(gt_interp, est_interp)
        
        # THEN results should be identical
        assert result1.ate_rmse == result2.ate_rmse
        assert result1.ate_mean == result2.ate_mean
        assert np.array_equal(result1.total_errors, result2.total_errors)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
