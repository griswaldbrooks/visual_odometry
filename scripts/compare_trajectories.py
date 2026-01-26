#!/usr/bin/env python3
"""Compare estimated and ground truth trajectories with time-series visualization.

Displays:
- 3D trajectory in viser
- X, Y, Z vs Time plots
- Error vs Time plot
- Time-matched ATE metrics

Usage:
    python scripts/compare_trajectories.py \
        --estimated trajectory.json \
        --ground-truth data/tum/.../groundtruth.txt \
        --estimated-fps 30
"""

import argparse
import numpy as np
import viser
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from trajectory_utils import (
    load_trajectory,
    compute_trajectory_error,
    TrajectoryInterpolator,
)


def compute_umeyama_alignment(
    gt_positions: np.ndarray, est_positions: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute optimal similarity transform using Umeyama algorithm.

    Finds scale, rotation, and translation that minimizes:
        sum || gt_i - (scale * R @ est_i + t) ||^2

    Args:
        gt_positions: Ground truth positions, shape (N, 3)
        est_positions: Estimated positions, shape (N, 3)

    Returns:
        Tuple of (scale, rotation, translation) where:
        - scale: float, uniform scale factor
        - rotation: (3, 3) array, rotation matrix
        - translation: (3,) array, translation vector
    """
    assert gt_positions.shape == est_positions.shape
    assert gt_positions.shape[1] == 3

    n = gt_positions.shape[0]

    # Center the points
    mu_gt = gt_positions.mean(axis=0)
    mu_est = est_positions.mean(axis=0)
    gt_centered = gt_positions - mu_gt
    est_centered = est_positions - mu_est

    # Compute variances
    sigma_gt_sq = (gt_centered ** 2).sum() / n
    sigma_est_sq = (est_centered ** 2).sum() / n

    # Compute cross-covariance matrix
    H = est_centered.T @ gt_centered / n

    # SVD of cross-covariance
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation
    R = Vt.T @ U.T

    # Handle reflection case (ensure proper rotation)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        S[-1] *= -1
        R = Vt.T @ U.T

    # Compute scale
    scale = S.sum() / sigma_est_sq

    # Compute translation
    translation = mu_gt - scale * R @ mu_est

    return scale, R, translation


def add_trajectory_to_scene(
    server: viser.ViserServer,
    interp: TrajectoryInterpolator,
    name: str,
    color: tuple[int, int, int],
    frustum_interval: int = 10,
    scale: float = 1.0,
) -> None:
    """Add a trajectory to the viser scene.

    Args:
        server: Viser server instance
        interp: Trajectory interpolator
        name: Name prefix for scene elements
        color: RGB color tuple
        frustum_interval: Add camera frustum every N frames
        scale: Scale factor to apply to positions
    """
    positions = interp.positions * scale

    # Check for sufficient points before drawing spline
    if len(positions) < 2:
        print(f"Warning: Trajectory has only {len(positions)} pose(s), skipping spline")
        return

    # Add trajectory line
    server.scene.add_spline_catmull_rom(
        f"{name}/trajectory",
        positions,
        color=color,
    )

    # Add camera frustums
    for i in range(0, len(interp), frustum_interval):
        pose = interp.interpolate(interp.timestamps[i])
        from scipy.spatial.transform import Rotation

        quat = pose.quaternion  # [qx, qy, qz, qw]
        wxyz = (quat[3], quat[0], quat[1], quat[2])
        scaled_position = pose.position * scale

        server.scene.add_camera_frustum(
            f"{name}/camera_{i:04d}",
            fov=60.0,
            aspect=1241 / 376,  # KITTI camera aspect ratio
            scale=0.15,
            wxyz=wxyz,
            position=tuple(scaled_position),
            color=color,
        )


def create_plot_html(
    gt_interp: TrajectoryInterpolator,
    est_interp: TrajectoryInterpolator,
    error_result,
    scale: float = 1.0,
) -> str:
    """Create HTML with plotly time-series plots and interactive scale slider.

    Args:
        gt_interp: Ground truth trajectory interpolator
        est_interp: Estimated trajectory interpolator
        error_result: Trajectory error result object
        scale: Initial scale factor applied to estimated trajectory

    Returns HTML string with embedded plotly charts and interactive scale control.
    """
    import json

    # Use full trajectories for plotting (not just aligned/interpolated)
    # Each trajectory's timestamps are relative to its own first timestamp
    # This handles GT with absolute timestamps (TUM) vs EST starting at 0

    # Ground truth: timestamps relative to first GT timestamp
    gt_t = gt_interp.timestamps - gt_interp.timestamps[0]
    gt_pos_full = gt_interp.positions

    # Estimated: timestamps relative to first EST timestamp
    est_t = est_interp.timestamps - est_interp.timestamps[0]
    est_pos_raw = est_interp.positions

    # GT interpolated at EST timestamps (for error computation and Umeyama alignment)
    gt_at_est, _ = gt_interp.interpolate_batch(est_interp.timestamps)

    # Compute optimal alignment using Umeyama algorithm
    # Uses GT interpolated at EST timestamps for point correspondence
    optimal_scale, optimal_rotation, optimal_translation = compute_umeyama_alignment(
        gt_at_est, est_pos_raw
    )

    # Convert arrays to JSON for JavaScript embedding
    gt_t_json = json.dumps(gt_t.tolist())
    gt_pos_x_json = json.dumps(gt_pos_full[:, 0].tolist())
    gt_pos_y_json = json.dumps(gt_pos_full[:, 1].tolist())
    gt_pos_z_json = json.dumps(gt_pos_full[:, 2].tolist())
    est_t_json = json.dumps(est_t.tolist())
    est_pos_x_json = json.dumps(est_pos_raw[:, 0].tolist())
    est_pos_y_json = json.dumps(est_pos_raw[:, 1].tolist())
    est_pos_z_json = json.dumps(est_pos_raw[:, 2].tolist())
    # GT interpolated at EST timestamps for error computation
    gt_at_est_x_json = json.dumps(gt_at_est[:, 0].tolist())
    gt_at_est_y_json = json.dumps(gt_at_est[:, 1].tolist())
    gt_at_est_z_json = json.dumps(gt_at_est[:, 2].tolist())

    # Compute trajectory extent (min/max) for ground truth and estimated
    # Use full trajectories, not just the interpolated/aligned portion
    import numpy as np
    gt_min = gt_interp.positions.min(axis=0)
    gt_max = gt_interp.positions.max(axis=0)
    est_min = est_pos_raw.min(axis=0)
    est_max = est_pos_raw.max(axis=0)

    # Create plotly figure data as JSON with interactive JavaScript
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 10px; background: #1a1a1a; color: #fff; }}
        .plot {{ width: 100%; height: 200px; margin-bottom: 10px; }}
        .stats {{ background: #2a2a2a; padding: 10px; border-radius: 5px; margin-bottom: 10px; }}
        h3 {{ margin: 5px 0; color: #4CAF50; }}
        .scale-control {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .scale-control label {{
            color: #ff9800;
            font-weight: bold;
            font-size: 16px;
        }}
        .scale-control input[type="range"] {{
            flex: 1;
            min-width: 200px;
            height: 8px;
            -webkit-appearance: none;
            background: #444;
            border-radius: 4px;
            outline: none;
        }}
        .scale-control input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #ff9800;
            border-radius: 50%;
            cursor: pointer;
        }}
        .scale-control input[type="range"]::-moz-range-thumb {{
            width: 20px;
            height: 20px;
            background: #ff9800;
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }}
        .scale-control input[type="number"] {{
            width: 80px;
            padding: 8px;
            background: #333;
            border: 1px solid #555;
            color: #fff;
            border-radius: 4px;
            font-size: 14px;
        }}
        .metrics-table {{
            border-collapse: collapse;
        }}
        .metrics-table td {{
            padding: 3px 10px 3px 0;
        }}
        .metrics-table td:first-child {{
            color: #aaa;
        }}
        .metrics-table td:last-child {{
            font-family: monospace;
        }}
        #rmse-value {{
            font-weight: bold;
            color: #4CAF50;
        }}
    </style>
</head>
<body>
    <div class="scale-control">
        <label for="scale-slider">Estimated Trajectory Scale:</label>
        <input type="range" id="scale-slider" min="0.001" max="10.0" step="0.001" value="{scale}">
        <input type="number" id="scale-input" min="0.001" max="10.0" step="0.001" value="{scale}">
    </div>

    <div class="scale-control">
        <label for="offset-x-slider">X Offset:</label>
        <input type="range" id="offset-x-slider" min="-100" max="100" step="0.01" value="0">
        <input type="number" id="offset-x-input" min="-100" max="100" step="0.01" value="0">
    </div>
    <div class="scale-control">
        <label for="offset-y-slider">Y Offset:</label>
        <input type="range" id="offset-y-slider" min="-100" max="100" step="0.01" value="0">
        <input type="number" id="offset-y-input" min="-100" max="100" step="0.01" value="0">
    </div>
    <div class="scale-control">
        <label for="offset-z-slider">Z Offset:</label>
        <input type="range" id="offset-z-slider" min="-100" max="100" step="0.01" value="0">
        <input type="number" id="offset-z-input" min="-100" max="100" step="0.01" value="0">
    </div>
    <div class="scale-control">
        <input type="checkbox" id="lock-offsets" style="width: 18px; height: 18px;">
        <label for="lock-offsets" style="color: #2196F3;">Lock offsets (change together by same delta)</label>
    </div>

    <div class="scale-control" style="background: #1a3a1a; border: 1px solid #4CAF50;">
        <button id="auto-align-btn" style="
            padding: 10px 20px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
        ">Auto-align (Umeyama)</button>
        <div style="margin-left: 20px;">
            <span style="color: #aaa;">Optimal values:</span>
            <span style="color: #4CAF50; font-family: monospace; margin-left: 10px;">
                scale={optimal_scale:.6f},
                offset=[{optimal_translation[0]:.4f}, {optimal_translation[1]:.4f}, {optimal_translation[2]:.4f}]
            </span>
            <br>
            <span style="color: #888; font-size: 12px;">(rotation ignored - UI only supports translation)</span>
        </div>
    </div>

    <div class="stats">
        <h3>Absolute Trajectory Error (ATE)</h3>
        <table class="metrics-table">
            <tr><td>RMSE:</td><td id="rmse-value">{error_result.ate_rmse:.4f} m</td></tr>
            <tr><td>Mean:</td><td id="mean-value">{error_result.ate_mean:.4f} m</td></tr>
            <tr><td>Median:</td><td id="median-value">{error_result.ate_median:.4f} m</td></tr>
            <tr><td>Std:</td><td id="std-value">{error_result.ate_std:.4f} m</td></tr>
            <tr><td>Min:</td><td id="min-value">{error_result.ate_min:.4f} m</td></tr>
            <tr><td>Max:</td><td id="max-value">{error_result.ate_max:.4f} m</td></tr>
        </table>
    </div>

    <div class="stats" style="display: flex; gap: 40px;">
        <div>
            <h3 style="color: #4CAF50;">Ground Truth Extent</h3>
            <table class="metrics-table">
                <tr><td>X:</td><td>[{gt_min[0]:.4f}, {gt_max[0]:.4f}] m</td></tr>
                <tr><td>Y:</td><td>[{gt_min[1]:.4f}, {gt_max[1]:.4f}] m</td></tr>
                <tr><td>Z:</td><td>[{gt_min[2]:.4f}, {gt_max[2]:.4f}] m</td></tr>
            </table>
        </div>
        <div>
            <h3 style="color: #ff6b6b;">Estimated Extent (scale=<span id="est-extent-scale">{scale}</span>, offset=[<span id="est-extent-offset">0, 0, 0</span>])</h3>
            <table class="metrics-table">
                <tr><td>X:</td><td id="est-x-extent">[{est_min[0]:.4f}, {est_max[0]:.4f}] m</td></tr>
                <tr><td>Y:</td><td id="est-y-extent">[{est_min[1]:.4f}, {est_max[1]:.4f}] m</td></tr>
                <tr><td>Z:</td><td id="est-z-extent">[{est_min[2]:.4f}, {est_max[2]:.4f}] m</td></tr>
            </table>
        </div>
    </div>

    <div id="plot_x" class="plot"></div>
    <div id="plot_y" class="plot"></div>
    <div id="plot_z" class="plot"></div>
    <div id="plot_error" class="plot"></div>

    <script>
        // Raw data arrays (unscaled) - GT and EST have separate timestamps
        var gtT = {gt_t_json};
        var gtPosX = {gt_pos_x_json};
        var gtPosY = {gt_pos_y_json};
        var gtPosZ = {gt_pos_z_json};
        var estT = {est_t_json};
        var estPosXRaw = {est_pos_x_json};
        var estPosYRaw = {est_pos_y_json};
        var estPosZRaw = {est_pos_z_json};
        // GT interpolated at EST timestamps (for error computation)
        var gtAtEstX = {gt_at_est_x_json};
        var gtAtEstY = {gt_at_est_y_json};
        var gtAtEstZ = {gt_at_est_z_json};

        // Optimal Umeyama alignment values (scale + translation, rotation ignored)
        var optimalScale = {optimal_scale};
        var optimalOffsetX = {optimal_translation[0]};
        var optimalOffsetY = {optimal_translation[1]};
        var optimalOffsetZ = {optimal_translation[2]};

        var layout = {{
            paper_bgcolor: '#1a1a1a',
            plot_bgcolor: '#1a1a1a',
            font: {{ color: '#fff' }},
            margin: {{ t: 30, r: 20, b: 40, l: 50 }},
            xaxis: {{ title: 'Time (s)', gridcolor: '#333' }},
            yaxis: {{ gridcolor: '#333' }},
            legend: {{ x: 0, y: 1 }}
        }};

        // Helper function to scale and offset an array
        function scaleArray(arr, scale) {{
            return arr.map(v => v * scale);
        }}

        // Helper function to scale and offset an array
        function scaleAndOffsetArray(arr, scale, offset) {{
            return arr.map(v => v * scale + offset);
        }}

        // Compute error metrics from GT (interpolated at EST times) and scaled+offset EST positions
        function computeErrors(scale, offsetX, offsetY, offsetZ) {{
            offsetX = offsetX || 0;
            offsetY = offsetY || 0;
            offsetZ = offsetZ || 0;
            var estX = scaleAndOffsetArray(estPosXRaw, scale, offsetX);
            var estY = scaleAndOffsetArray(estPosYRaw, scale, offsetY);
            var estZ = scaleAndOffsetArray(estPosZRaw, scale, offsetZ);

            var errors = [];
            var errX = [];
            var errY = [];
            var errZ = [];

            // Use GT interpolated at EST timestamps for error computation
            for (var i = 0; i < gtAtEstX.length; i++) {{
                var dx = gtAtEstX[i] - estX[i];
                var dy = gtAtEstY[i] - estY[i];
                var dz = gtAtEstZ[i] - estZ[i];
                errX.push(Math.abs(dx));
                errY.push(Math.abs(dy));
                errZ.push(Math.abs(dz));
                errors.push(Math.sqrt(dx*dx + dy*dy + dz*dz));
            }}

            // Compute statistics
            var sum = errors.reduce((a, b) => a + b, 0);
            var mean = sum / errors.length;
            var sqSum = errors.reduce((a, b) => a + b*b, 0);
            var rmse = Math.sqrt(sqSum / errors.length);
            var sorted = [...errors].sort((a, b) => a - b);
            var median = sorted[Math.floor(sorted.length / 2)];
            var variance = errors.reduce((a, b) => a + (b - mean) ** 2, 0) / errors.length;
            var std = Math.sqrt(variance);
            var minErr = Math.min(...errors);
            var maxErr = Math.max(...errors);

            return {{
                total: errors,
                x: errX,
                y: errY,
                z: errZ,
                rmse: rmse,
                mean: mean,
                median: median,
                std: std,
                min: minErr,
                max: maxErr
            }};
        }}

        // Initialize plots with initial scale
        var initialScale = {scale};

        function initPlots(scale, offsetX, offsetY, offsetZ) {{
            offsetX = offsetX || 0;
            offsetY = offsetY || 0;
            offsetZ = offsetZ || 0;
            var estX = scaleAndOffsetArray(estPosXRaw, scale, offsetX);
            var estY = scaleAndOffsetArray(estPosYRaw, scale, offsetY);
            var estZ = scaleAndOffsetArray(estPosZRaw, scale, offsetZ);
            var errData = computeErrors(scale, offsetX, offsetY, offsetZ);

            // X vs Time - GT at native timestamps, EST at its timestamps
            Plotly.newPlot('plot_x', [
                {{ x: gtT, y: gtPosX, name: 'GT X', line: {{ color: '#4CAF50' }} }},
                {{ x: estT, y: estX, name: 'Est X', line: {{ color: '#f44336' }} }}
            ], {{...layout, yaxis: {{...layout.yaxis, title: 'X (m)'}}}});

            // Y vs Time
            Plotly.newPlot('plot_y', [
                {{ x: gtT, y: gtPosY, name: 'GT Y', line: {{ color: '#4CAF50' }} }},
                {{ x: estT, y: estY, name: 'Est Y', line: {{ color: '#f44336' }} }}
            ], {{...layout, yaxis: {{...layout.yaxis, title: 'Y (m)'}}}});

            // Z vs Time
            Plotly.newPlot('plot_z', [
                {{ x: gtT, y: gtPosZ, name: 'GT Z', line: {{ color: '#4CAF50' }} }},
                {{ x: estT, y: estZ, name: 'Est Z', line: {{ color: '#f44336' }} }}
            ], {{...layout, yaxis: {{...layout.yaxis, title: 'Z (m)'}}}});

            // Error vs Time - computed at EST timestamps
            Plotly.newPlot('plot_error', [
                {{ x: estT, y: errData.total, name: 'Total Error', line: {{ color: '#ff9800' }}, fill: 'tozeroy' }},
                {{ x: estT, y: errData.x, name: 'X Error', line: {{ color: '#2196F3', dash: 'dot' }} }},
                {{ x: estT, y: errData.y, name: 'Y Error', line: {{ color: '#9C27B0', dash: 'dot' }} }},
                {{ x: estT, y: errData.z, name: 'Z Error', line: {{ color: '#00BCD4', dash: 'dot' }} }}
            ], {{...layout, yaxis: {{...layout.yaxis, title: 'Error (m)'}}}});
        }}

        // Update plots when scale or offsets change
        function updatePlots(scale, offsetX, offsetY, offsetZ) {{
            offsetX = offsetX || 0;
            offsetY = offsetY || 0;
            offsetZ = offsetZ || 0;
            var estX = scaleAndOffsetArray(estPosXRaw, scale, offsetX);
            var estY = scaleAndOffsetArray(estPosYRaw, scale, offsetY);
            var estZ = scaleAndOffsetArray(estPosZRaw, scale, offsetZ);
            var errData = computeErrors(scale, offsetX, offsetY, offsetZ);

            // Update X plot (trace index 1 is estimated)
            Plotly.restyle('plot_x', {{ y: [estX] }}, [1]);

            // Update Y plot
            Plotly.restyle('plot_y', {{ y: [estY] }}, [1]);

            // Update Z plot
            Plotly.restyle('plot_z', {{ y: [estZ] }}, [1]);

            // Update Error plot (all 4 traces)
            Plotly.restyle('plot_error', {{ y: [errData.total, errData.x, errData.y, errData.z] }}, [0, 1, 2, 3]);

            // Update metrics display
            document.getElementById('rmse-value').textContent = errData.rmse.toFixed(4) + ' m';
            document.getElementById('mean-value').textContent = errData.mean.toFixed(4) + ' m';
            document.getElementById('median-value').textContent = errData.median.toFixed(4) + ' m';
            document.getElementById('std-value').textContent = errData.std.toFixed(4) + ' m';
            document.getElementById('min-value').textContent = errData.min.toFixed(4) + ' m';
            document.getElementById('max-value').textContent = errData.max.toFixed(4) + ' m';

            // Update estimated extent display
            var estXMin = Math.min(...estX).toFixed(4);
            var estXMax = Math.max(...estX).toFixed(4);
            var estYMin = Math.min(...estY).toFixed(4);
            var estYMax = Math.max(...estY).toFixed(4);
            var estZMin = Math.min(...estZ).toFixed(4);
            var estZMax = Math.max(...estZ).toFixed(4);
            document.getElementById('est-x-extent').textContent = '[' + estXMin + ', ' + estXMax + '] m';
            document.getElementById('est-y-extent').textContent = '[' + estYMin + ', ' + estYMax + '] m';
            document.getElementById('est-z-extent').textContent = '[' + estZMin + ', ' + estZMax + '] m';
            document.getElementById('est-extent-scale').textContent = scale.toFixed(3);
            document.getElementById('est-extent-offset').textContent = offsetX.toFixed(2) + ', ' + offsetY.toFixed(2) + ', ' + offsetZ.toFixed(2);
        }}

        // Initialize
        initPlots(initialScale, 0, 0, 0);

        // Scale slider event listeners
        var scaleSlider = document.getElementById('scale-slider');
        var scaleInput = document.getElementById('scale-input');

        // Offset slider and input elements
        var offsetXSlider = document.getElementById('offset-x-slider');
        var offsetXInput = document.getElementById('offset-x-input');
        var offsetYSlider = document.getElementById('offset-y-slider');
        var offsetYInput = document.getElementById('offset-y-input');
        var offsetZSlider = document.getElementById('offset-z-slider');
        var offsetZInput = document.getElementById('offset-z-input');
        var lockOffsetsCheckbox = document.getElementById('lock-offsets');

        // Track previous offset values for delta computation when locked
        var prevOffsetX = 0;
        var prevOffsetY = 0;
        var prevOffsetZ = 0;

        // Helper to get current offsets
        function getOffsets() {{
            return {{
                x: parseFloat(offsetXSlider.value),
                y: parseFloat(offsetYSlider.value),
                z: parseFloat(offsetZSlider.value)
            }};
        }}

        // Helper to update all offset UI elements
        function setOffsets(x, y, z) {{
            // Clamp values to valid range
            x = Math.max(-100, Math.min(100, x));
            y = Math.max(-100, Math.min(100, y));
            z = Math.max(-100, Math.min(100, z));

            offsetXSlider.value = x;
            offsetXInput.value = x.toFixed(2);
            offsetYSlider.value = y;
            offsetYInput.value = y.toFixed(2);
            offsetZSlider.value = z;
            offsetZInput.value = z.toFixed(2);

            // Update previous values
            prevOffsetX = x;
            prevOffsetY = y;
            prevOffsetZ = z;
        }}

        // Helper to trigger plot update with current scale and offsets
        function triggerUpdate() {{
            var scale = parseFloat(scaleSlider.value);
            var offsets = getOffsets();
            updatePlots(scale, offsets.x, offsets.y, offsets.z);
        }}

        // Handle offset change with locking logic
        function handleOffsetChange(changedAxis, newValue) {{
            var locked = lockOffsetsCheckbox.checked;
            var offsets = getOffsets();

            if (locked) {{
                // Compute delta from previous value
                var delta = 0;
                if (changedAxis === 'x') {{
                    delta = newValue - prevOffsetX;
                }} else if (changedAxis === 'y') {{
                    delta = newValue - prevOffsetY;
                }} else if (changedAxis === 'z') {{
                    delta = newValue - prevOffsetZ;
                }}

                // Apply delta to all offsets
                var newX = prevOffsetX + delta;
                var newY = prevOffsetY + delta;
                var newZ = prevOffsetZ + delta;

                setOffsets(newX, newY, newZ);
            }} else {{
                // Just update the changed axis and track previous
                if (changedAxis === 'x') {{
                    prevOffsetX = newValue;
                    offsetXInput.value = newValue.toFixed(2);
                }} else if (changedAxis === 'y') {{
                    prevOffsetY = newValue;
                    offsetYInput.value = newValue.toFixed(2);
                }} else if (changedAxis === 'z') {{
                    prevOffsetZ = newValue;
                    offsetZInput.value = newValue.toFixed(2);
                }}
            }}

            triggerUpdate();
        }}

        scaleSlider.addEventListener('input', function() {{
            var scale = parseFloat(this.value);
            scaleInput.value = scale.toFixed(3);
            triggerUpdate();
        }});

        scaleInput.addEventListener('change', function() {{
            var scale = parseFloat(this.value);
            if (scale >= 0.001 && scale <= 10.0) {{
                scaleSlider.value = scale;
                triggerUpdate();
            }}
        }});

        // Offset X event listeners
        offsetXSlider.addEventListener('input', function() {{
            handleOffsetChange('x', parseFloat(this.value));
        }});
        offsetXInput.addEventListener('change', function() {{
            var val = parseFloat(this.value);
            if (val >= -100 && val <= 100) {{
                offsetXSlider.value = val;
                handleOffsetChange('x', val);
            }}
        }});

        // Offset Y event listeners
        offsetYSlider.addEventListener('input', function() {{
            handleOffsetChange('y', parseFloat(this.value));
        }});
        offsetYInput.addEventListener('change', function() {{
            var val = parseFloat(this.value);
            if (val >= -100 && val <= 100) {{
                offsetYSlider.value = val;
                handleOffsetChange('y', val);
            }}
        }});

        // Offset Z event listeners
        offsetZSlider.addEventListener('input', function() {{
            handleOffsetChange('z', parseFloat(this.value));
        }});
        offsetZInput.addEventListener('change', function() {{
            var val = parseFloat(this.value);
            if (val >= -100 && val <= 100) {{
                offsetZSlider.value = val;
                handleOffsetChange('z', val);
            }}
        }});

        // Auto-align button click handler
        document.getElementById('auto-align-btn').addEventListener('click', function() {{
            // Apply optimal scale
            scaleSlider.value = optimalScale;
            scaleInput.value = optimalScale.toFixed(6);

            // Apply optimal offsets (translation from Umeyama)
            setOffsets(optimalOffsetX, optimalOffsetY, optimalOffsetZ);

            // Trigger plot update
            triggerUpdate();
        }});
    </script>
</body>
</html>
"""
    return html


def compute_scaled_error(
    gt_interp: TrajectoryInterpolator,
    est_interp: TrajectoryInterpolator,
    scale: float,
):
    """Compute trajectory error with scaled estimated positions.

    Args:
        gt_interp: Ground truth trajectory interpolator
        est_interp: Estimated trajectory interpolator
        scale: Scale factor to apply to estimated positions

    Returns:
        TrajectoryErrorResult with errors computed on scaled trajectory
    """
    # Create a scaled version of the estimated interpolator
    scaled_positions = est_interp.positions * scale
    scaled_interp = TrajectoryInterpolator(
        est_interp.timestamps,
        scaled_positions,
        est_interp.quaternions,
    )
    return compute_trajectory_error(gt_interp, scaled_interp)


def remove_trajectory_from_scene(
    server: viser.ViserServer,
    interp: TrajectoryInterpolator,
    name: str,
    frustum_interval: int = 10,
) -> None:
    """Remove a trajectory from the viser scene.

    Args:
        server: Viser server instance
        interp: Trajectory interpolator (to know how many frustums were added)
        name: Name prefix for scene elements
        frustum_interval: Interval used when adding frustums
    """
    # Remove trajectory spline
    try:
        server.scene.remove_by_name(f"{name}/trajectory")
    except Exception:
        pass

    # Remove camera frustums
    for i in range(0, len(interp), frustum_interval):
        try:
            server.scene.remove_by_name(f"{name}/camera_{i:04d}")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare estimated and ground truth trajectories"
    )
    parser.add_argument(
        "--estimated",
        type=Path,
        required=True,
        help="Path to estimated trajectory (trajectory.json or TUM/KITTI format)",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to ground truth trajectory (TUM or KITTI format)",
    )
    parser.add_argument(
        "--estimated-fps",
        type=float,
        default=10.0,
        help="FPS for estimated trajectory (for formats without timestamps)",
    )
    parser.add_argument(
        "--ground-truth-fps",
        type=float,
        default=10.0,
        help="FPS for ground truth (for KITTI format)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to visualize",
    )
    args = parser.parse_args()

    # Check file existence before loading
    if not args.estimated.exists():
        print(f"Error: Estimated trajectory not found: {args.estimated}")
        sys.exit(1)

    if not args.ground_truth.exists():
        print(f"Error: Ground truth not found: {args.ground_truth}")
        sys.exit(1)

    # Load trajectories
    print(f"Loading estimated trajectory from {args.estimated}")
    est_interp = load_trajectory(args.estimated, fps=args.estimated_fps)
    print(f"  Loaded {len(est_interp)} poses, duration: {est_interp.duration:.2f}s")

    print(f"Loading ground truth from {args.ground_truth}")
    gt_interp = load_trajectory(args.ground_truth, fps=args.ground_truth_fps)
    print(f"  Loaded {len(gt_interp)} poses, duration: {gt_interp.duration:.2f}s")

    # Limit frames if requested
    if args.max_frames:
        # Create new interpolators with limited data
        n_est = min(args.max_frames, len(est_interp))
        n_gt = min(args.max_frames, len(gt_interp))

        est_interp = TrajectoryInterpolator(
            est_interp.timestamps[:n_est],
            est_interp.positions[:n_est],
            est_interp.quaternions[:n_est],
        )
        gt_interp = TrajectoryInterpolator(
            gt_interp.timestamps[:n_gt],
            gt_interp.positions[:n_gt],
            gt_interp.quaternions[:n_gt],
        )

    # Start viser server
    server = viser.ViserServer()
    print(f"\nViser server started at: http://localhost:{server.get_port()}")

    # Add world frame
    server.scene.add_frame("world", wxyz=(1, 0, 0, 0), position=(0, 0, 0), axes_length=1.0)

    # Add GUI controls for scale
    scale_slider = server.gui.add_slider(
        "Estimated Scale",
        min=0.001,
        max=10.0,
        step=0.001,
        initial_value=1.0,
    )

    # Add a text display for current error metrics
    error_text = server.gui.add_text(
        "ATE Metrics",
        initial_value="Computing...",
        disabled=True,
    )

    # Track current scale for updates
    current_scale = [1.0]  # Use list to allow mutation in callback

    # Import tempfile and webbrowser for plots
    import tempfile
    import webbrowser

    # Track plots file path
    plots_path = [None]

    def update_visualization(scale: float) -> None:
        """Update the visualization with a new scale factor."""
        current_scale[0] = scale

        # Remove old estimated trajectory
        remove_trajectory_from_scene(server, est_interp, "est")

        # Add scaled estimated trajectory
        add_trajectory_to_scene(server, est_interp, "est", (255, 0, 0), scale=scale)

        # Compute scaled error
        error_result = compute_scaled_error(gt_interp, est_interp, scale)

        # Update error text display
        error_text.value = (
            f"RMSE: {error_result.ate_rmse:.4f} m | "
            f"Mean: {error_result.ate_mean:.4f} m | "
            f"Median: {error_result.ate_median:.4f} m"
        )

        # Update plots HTML
        plots_html = create_plot_html(gt_interp, est_interp, error_result, scale=scale)

        # Write updated plots
        if plots_path[0]:
            with open(plots_path[0], "w") as f:
                f.write(plots_html)
            print(f"  Scale: {scale:.3f} | ATE RMSE: {error_result.ate_rmse:.4f} m (refresh browser for updated plots)")
        else:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
                f.write(plots_html)
                plots_path[0] = f.name

        return error_result

    # Register callback for scale changes
    @scale_slider.on_update
    def on_scale_change(event: viser.GuiEvent) -> None:
        update_visualization(scale_slider.value)

    # Add trajectories to 3D scene
    print("Adding trajectories to 3D scene...")
    add_trajectory_to_scene(server, gt_interp, "gt", (0, 255, 0))  # Green

    # Initial visualization with scale=1.0
    print("\nComputing initial trajectory error...")
    error_result = update_visualization(1.0)

    print(f"\nAbsolute Trajectory Error (ATE) at scale=1.0:")
    print(f"  RMSE:   {error_result.ate_rmse:.4f} m")
    print(f"  Mean:   {error_result.ate_mean:.4f} m")
    print(f"  Median: {error_result.ate_median:.4f} m")
    print(f"  Std:    {error_result.ate_std:.4f} m")
    print(f"  Min:    {error_result.ate_min:.4f} m")
    print(f"  Max:    {error_result.ate_max:.4f} m")

    # Open plots in browser
    print(f"\nTime-series plots saved to: {plots_path[0]}")
    print(f"Opening plots in browser...")
    webbrowser.open(f"file://{plots_path[0]}")

    print("\n3D visualization: http://localhost:8080")
    print("  Green = Ground Truth")
    print("  Red = Estimated (scalable)")
    print("\nUse the 'Estimated Scale' slider in the viser GUI to adjust scale.")
    print("Refresh the browser plots page to see updated metrics after scale change.")
    print("\nPress Ctrl+C to exit.")

    import time
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
