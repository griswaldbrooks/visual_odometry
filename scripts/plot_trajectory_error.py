#!/usr/bin/env python3
"""Standalone HTML trajectory error visualization tool.

Generates an interactive HTML file with:
- X, Y, Z vs Time plots
- Error vs Time plot
- Interactive scale and offset controls
- Auto-align button (Umeyama algorithm)
- Real-time error metrics

Usage:
    python scripts/plot_trajectory_error.py <estimated> <ground_truth>

Examples:
    python scripts/plot_trajectory_error.py trajectory.json data/tum/.../groundtruth.txt
    python scripts/plot_trajectory_error.py output.json data/kitti/.../poses.txt
"""

import argparse
import json
import sys
import tempfile
import webbrowser
from pathlib import Path

import numpy as np

# Add scripts directory to path for trajectory_utils import
sys.path.insert(0, str(Path(__file__).parent))

from trajectory_utils import (
    load_trajectory,
    compute_trajectory_error,
    TrajectoryInterpolator,
    TrajectoryErrorResult,
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


def create_plot_html(
    gt_interp: TrajectoryInterpolator,
    est_interp: TrajectoryInterpolator,
    error_result: TrajectoryErrorResult,
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
        .axis-control {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .axis-control label {{
            color: #9C27B0;
            font-weight: bold;
            font-size: 14px;
        }}
        .axis-control select {{
            padding: 8px 12px;
            background: #333;
            border: 1px solid #555;
            color: #fff;
            border-radius: 4px;
            font-size: 14px;
            cursor: pointer;
        }}
        .axis-control select:hover {{
            border-color: #9C27B0;
        }}
        .axis-group {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-right: 20px;
        }}
        .axis-group span {{
            color: #aaa;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="axis-control" style="background: #1a2a3a; border: 1px solid #9C27B0;">
        <label style="color: #9C27B0; font-size: 16px;">Axis Remapping (Estimated Trajectory):</label>
        <div class="axis-group">
            <span>X from:</span>
            <select id="axis-x-select">
                <option value="X" selected>X</option>
                <option value="-X">-X</option>
                <option value="Y">Y</option>
                <option value="-Y">-Y</option>
                <option value="Z">Z</option>
                <option value="-Z">-Z</option>
            </select>
        </div>
        <div class="axis-group">
            <span>Y from:</span>
            <select id="axis-y-select">
                <option value="X">X</option>
                <option value="-X">-X</option>
                <option value="Y" selected>Y</option>
                <option value="-Y">-Y</option>
                <option value="Z">Z</option>
                <option value="-Z">-Z</option>
            </select>
        </div>
        <div class="axis-group">
            <span>Z from:</span>
            <select id="axis-z-select">
                <option value="X">X</option>
                <option value="-X">-X</option>
                <option value="Y">Y</option>
                <option value="-Y">-Y</option>
                <option value="Z" selected>Z</option>
                <option value="-Z">-Z</option>
            </select>
        </div>
        <button id="reset-axes-btn" style="
            padding: 6px 12px;
            background: #555;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
        ">Reset to Default</button>
        <button id="swap-yz-btn" style="
            padding: 6px 12px;
            background: #9C27B0;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
        ">Swap Y/Z (Y-up to Z-up)</button>
        <button id="auto-find-axes-btn" style="
            padding: 6px 12px;
            background: #FF5722;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
        ">Auto-find Best Axes</button>
    </div>

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
            <span id="optimal-values" style="color: #4CAF50; font-family: monospace; margin-left: 10px;">
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
            <div style="color: #9C27B0; font-size: 12px; margin-bottom: 5px;">Axis mapping: <span id="est-axis-mapping">X->X, Y->Y, Z->Z</span></div>
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

        // Axis remapping configuration
        var axisMapping = {{
            x: {{ source: 'X', negate: false }},
            y: {{ source: 'Y', negate: false }},
            z: {{ source: 'Z', negate: false }}
        }};

        // Parse axis selection (e.g., "-Y" -> {{ source: 'Y', negate: true }})
        function parseAxisSelection(value) {{
            if (value.startsWith('-')) {{
                return {{ source: value.substring(1), negate: true }};
            }}
            return {{ source: value, negate: false }};
        }}

        // Get source array for axis
        function getSourceArray(axis) {{
            switch(axis) {{
                case 'X': return estPosXRaw;
                case 'Y': return estPosYRaw;
                case 'Z': return estPosZRaw;
                default: return estPosXRaw;
            }}
        }}

        // Apply axis remapping to get remapped X, Y, Z arrays
        function getRemappedEstimated() {{
            var xMapping = axisMapping.x;
            var yMapping = axisMapping.y;
            var zMapping = axisMapping.z;

            var srcX = getSourceArray(xMapping.source);
            var srcY = getSourceArray(yMapping.source);
            var srcZ = getSourceArray(zMapping.source);

            var remappedX = xMapping.negate ? srcX.map(v => -v) : srcX.slice();
            var remappedY = yMapping.negate ? srcY.map(v => -v) : srcY.slice();
            var remappedZ = zMapping.negate ? srcZ.map(v => -v) : srcZ.slice();

            return {{ x: remappedX, y: remappedY, z: remappedZ }};
        }}

        // Apply scale and offset to remapped arrays
        function getTransformedEstimated(scale, offsetX, offsetY, offsetZ) {{
            var remapped = getRemappedEstimated();
            return {{
                x: remapped.x.map(v => v * scale + offsetX),
                y: remapped.y.map(v => v * scale + offsetY),
                z: remapped.z.map(v => v * scale + offsetZ)
            }};
        }}

        // Compute error metrics from GT (interpolated at EST times) and scaled+offset EST positions
        function computeErrors(scale, offsetX, offsetY, offsetZ) {{
            offsetX = offsetX || 0;
            offsetY = offsetY || 0;
            offsetZ = offsetZ || 0;

            // Use axis-remapped estimated trajectory
            var transformed = getTransformedEstimated(scale, offsetX, offsetY, offsetZ);
            var estX = transformed.x;
            var estY = transformed.y;
            var estZ = transformed.z;

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

            // Use axis-remapped estimated trajectory
            var transformed = getTransformedEstimated(scale, offsetX, offsetY, offsetZ);
            var estX = transformed.x;
            var estY = transformed.y;
            var estZ = transformed.z;
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

            // Use axis-remapped estimated trajectory
            var transformed = getTransformedEstimated(scale, offsetX, offsetY, offsetZ);
            var estX = transformed.x;
            var estY = transformed.y;
            var estZ = transformed.z;
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

            // Update axis mapping display
            var xMap = (axisMapping.x.negate ? '-' : '') + axisMapping.x.source;
            var yMap = (axisMapping.y.negate ? '-' : '') + axisMapping.y.source;
            var zMap = (axisMapping.z.negate ? '-' : '') + axisMapping.z.source;
            document.getElementById('est-axis-mapping').textContent = 'X->' + xMap + ', Y->' + yMap + ', Z->' + zMap;
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

        // Axis remapping controls
        var axisXSelect = document.getElementById('axis-x-select');
        var axisYSelect = document.getElementById('axis-y-select');
        var axisZSelect = document.getElementById('axis-z-select');
        var resetAxesBtn = document.getElementById('reset-axes-btn');
        var swapYZBtn = document.getElementById('swap-yz-btn');

        // Recompute Umeyama alignment based on current axis mapping
        function recomputeUmeyamaAlignment() {{
            // Get remapped estimated positions (before scale/offset)
            var remapped = getRemappedEstimated();
            var n = remapped.x.length;

            // Compute centroids
            var muGtX = gtAtEstX.reduce((a, b) => a + b, 0) / n;
            var muGtY = gtAtEstY.reduce((a, b) => a + b, 0) / n;
            var muGtZ = gtAtEstZ.reduce((a, b) => a + b, 0) / n;

            var muEstX = remapped.x.reduce((a, b) => a + b, 0) / n;
            var muEstY = remapped.y.reduce((a, b) => a + b, 0) / n;
            var muEstZ = remapped.z.reduce((a, b) => a + b, 0) / n;

            // Compute variance of estimated (after remapping)
            var sigmaEstSq = 0;
            for (var i = 0; i < n; i++) {{
                var dx = remapped.x[i] - muEstX;
                var dy = remapped.y[i] - muEstY;
                var dz = remapped.z[i] - muEstZ;
                sigmaEstSq += dx * dx + dy * dy + dz * dz;
            }}
            sigmaEstSq /= n;

            // Compute variance of ground truth
            var sigmaGtSq = 0;
            for (var i = 0; i < n; i++) {{
                var dx = gtAtEstX[i] - muGtX;
                var dy = gtAtEstY[i] - muGtY;
                var dz = gtAtEstZ[i] - muGtZ;
                sigmaGtSq += dx * dx + dy * dy + dz * dz;
            }}
            sigmaGtSq /= n;

            // Scale is approximately sqrt(sigmaGt/sigmaEst)
            var newScale = sigmaEstSq > 1e-10 ? Math.sqrt(sigmaGtSq / sigmaEstSq) : 1.0;

            // Translation (ignoring rotation for UI simplicity)
            var newOffsetX = muGtX - newScale * muEstX;
            var newOffsetY = muGtY - newScale * muEstY;
            var newOffsetZ = muGtZ - newScale * muEstZ;

            // Update optimal values
            optimalScale = newScale;
            optimalOffsetX = newOffsetX;
            optimalOffsetY = newOffsetY;
            optimalOffsetZ = newOffsetZ;

            // Update display of optimal values
            var optimalDisplay = document.querySelector('#optimal-values');
            if (optimalDisplay) {{
                optimalDisplay.innerHTML = 'scale=' + newScale.toFixed(6) +
                    ', offset=[' + newOffsetX.toFixed(4) + ', ' + newOffsetY.toFixed(4) + ', ' + newOffsetZ.toFixed(4) + ']';
            }}

            console.log('Umeyama recomputed:', {{ scale: newScale, offset: [newOffsetX, newOffsetY, newOffsetZ] }});
        }}

        // Update axis mapping and trigger plot update
        function updateAxisMapping() {{
            axisMapping.x = parseAxisSelection(axisXSelect.value);
            axisMapping.y = parseAxisSelection(axisYSelect.value);
            axisMapping.z = parseAxisSelection(axisZSelect.value);

            console.log('Axis mapping updated:', axisMapping);

            // Recompute Umeyama alignment with new axis mapping
            recomputeUmeyamaAlignment();

            // Trigger plot update
            triggerUpdate();
        }}

        // Event listeners for axis dropdowns
        axisXSelect.addEventListener('change', updateAxisMapping);
        axisYSelect.addEventListener('change', updateAxisMapping);
        axisZSelect.addEventListener('change', updateAxisMapping);

        // Reset axes to default
        resetAxesBtn.addEventListener('click', function() {{
            axisXSelect.value = 'X';
            axisYSelect.value = 'Y';
            axisZSelect.value = 'Z';
            updateAxisMapping();
        }});

        // Swap Y and Z (common Y-up to Z-up conversion)
        swapYZBtn.addEventListener('click', function() {{
            axisXSelect.value = 'X';
            axisYSelect.value = 'Z';
            axisZSelect.value = '-Y';
            updateAxisMapping();
        }});

        // Auto-find best axis configuration
        var autoFindAxesBtn = document.getElementById('auto-find-axes-btn');

        function applyAxisConfigAndComputeError(xMap, yMap, zMap) {{
            // Parse axis mappings
            function parseAxis(value) {{
                if (value.startsWith('-')) {{
                    return {{ source: value.substring(1), negate: true }};
                }}
                return {{ source: value, negate: false }};
            }}

            var xParsed = parseAxis(xMap);
            var yParsed = parseAxis(yMap);
            var zParsed = parseAxis(zMap);

            // Get source arrays
            function getSrc(axis) {{
                switch(axis) {{
                    case 'X': return estPosXRaw;
                    case 'Y': return estPosYRaw;
                    case 'Z': return estPosZRaw;
                }}
            }}

            var srcX = getSrc(xParsed.source);
            var srcY = getSrc(yParsed.source);
            var srcZ = getSrc(zParsed.source);

            // Apply remapping
            var remappedX = xParsed.negate ? srcX.map(v => -v) : srcX.slice();
            var remappedY = yParsed.negate ? srcY.map(v => -v) : srcY.slice();
            var remappedZ = zParsed.negate ? srcZ.map(v => -v) : srcZ.slice();

            var n = remappedX.length;

            // Compute centroids
            var muGtX = gtAtEstX.reduce((a, b) => a + b, 0) / n;
            var muGtY = gtAtEstY.reduce((a, b) => a + b, 0) / n;
            var muGtZ = gtAtEstZ.reduce((a, b) => a + b, 0) / n;
            var muEstX = remappedX.reduce((a, b) => a + b, 0) / n;
            var muEstY = remappedY.reduce((a, b) => a + b, 0) / n;
            var muEstZ = remappedZ.reduce((a, b) => a + b, 0) / n;

            // Compute variances
            var sigmaEstSq = 0, sigmaGtSq = 0;
            for (var i = 0; i < n; i++) {{
                var dex = remappedX[i] - muEstX, dey = remappedY[i] - muEstY, dez = remappedZ[i] - muEstZ;
                var dgx = gtAtEstX[i] - muGtX, dgy = gtAtEstY[i] - muGtY, dgz = gtAtEstZ[i] - muGtZ;
                sigmaEstSq += dex*dex + dey*dey + dez*dez;
                sigmaGtSq += dgx*dgx + dgy*dgy + dgz*dgz;
            }}
            sigmaEstSq /= n;
            sigmaGtSq /= n;

            // Compute scale and offset
            var scale = sigmaEstSq > 1e-10 ? Math.sqrt(sigmaGtSq / sigmaEstSq) : 1.0;
            var offsetX = muGtX - scale * muEstX;
            var offsetY = muGtY - scale * muEstY;
            var offsetZ = muGtZ - scale * muEstZ;

            // Compute RMSE with this alignment
            var sqSum = 0;
            for (var i = 0; i < n; i++) {{
                var ex = gtAtEstX[i] - (remappedX[i] * scale + offsetX);
                var ey = gtAtEstY[i] - (remappedY[i] * scale + offsetY);
                var ez = gtAtEstZ[i] - (remappedZ[i] * scale + offsetZ);
                sqSum += ex*ex + ey*ey + ez*ez;
            }}
            var rmse = Math.sqrt(sqSum / n);

            return {{ rmse: rmse, scale: scale, offset: [offsetX, offsetY, offsetZ] }};
        }}

        function findBestAxisConfiguration() {{
            var axes = ['X', 'Y', 'Z'];
            var signs = ['', '-'];
            var bestRmse = Infinity;
            var bestConfig = ['X', 'Y', 'Z'];
            var bestScale = 1.0;
            var bestOffset = [0, 0, 0];

            // Generate all permutations of axes
            var perms = [
                ['X', 'Y', 'Z'], ['X', 'Z', 'Y'], ['Y', 'X', 'Z'],
                ['Y', 'Z', 'X'], ['Z', 'X', 'Y'], ['Z', 'Y', 'X']
            ];

            // Try all 6 permutations × 8 sign combinations = 48 configurations
            for (var p = 0; p < perms.length; p++) {{
                var perm = perms[p];
                for (var sx = 0; sx < 2; sx++) {{
                    for (var sy = 0; sy < 2; sy++) {{
                        for (var sz = 0; sz < 2; sz++) {{
                            var xMap = signs[sx] + perm[0];
                            var yMap = signs[sy] + perm[1];
                            var zMap = signs[sz] + perm[2];

                            var result = applyAxisConfigAndComputeError(xMap, yMap, zMap);

                            if (result.rmse < bestRmse) {{
                                bestRmse = result.rmse;
                                bestConfig = [xMap || perm[0], yMap || perm[1], zMap || perm[2]];
                                bestScale = result.scale;
                                bestOffset = result.offset;
                            }}
                        }}
                    }}
                }}
            }}

            return {{ config: bestConfig, scale: bestScale, offset: bestOffset, rmse: bestRmse }};
        }}

        autoFindAxesBtn.addEventListener('click', function() {{
            console.log('Searching for best axis configuration (48 combinations)...');
            var best = findBestAxisConfiguration();

            // Apply best configuration
            axisXSelect.value = best.config[0];
            axisYSelect.value = best.config[1];
            axisZSelect.value = best.config[2];

            // Update axis mapping
            axisMapping.x = parseAxisSelection(best.config[0]);
            axisMapping.y = parseAxisSelection(best.config[1]);
            axisMapping.z = parseAxisSelection(best.config[2]);

            // Apply scale and offset
            scaleSlider.value = best.scale;
            scaleInput.value = best.scale.toFixed(6);
            setOffsets(best.offset[0], best.offset[1], best.offset[2]);

            // Update optimal values
            optimalScale = best.scale;
            optimalOffsetX = best.offset[0];
            optimalOffsetY = best.offset[1];
            optimalOffsetZ = best.offset[2];

            // Update display
            var optimalDisplay = document.querySelector('#optimal-values');
            if (optimalDisplay) {{
                optimalDisplay.innerHTML = 'scale=' + best.scale.toFixed(6) +
                    ', offset=[' + best.offset[0].toFixed(4) + ', ' + best.offset[1].toFixed(4) + ', ' + best.offset[2].toFixed(4) + ']';
            }}

            // Trigger update
            triggerUpdate();

            console.log('Best axes: [' + best.config.join(', ') + '], scale=' + best.scale.toFixed(6) + ', RMSE=' + best.rmse.toFixed(4) + 'm');
        }});
    </script>
</body>
</html>
"""
    return html


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML trajectory error visualization",
        epilog=(
            "Examples:\n"
            "  %(prog)s trajectory.json data/tum/.../groundtruth.txt\n"
            "  %(prog)s output.json data/kitti/.../poses.txt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "estimated",
        type=Path,
        help="Path to estimated trajectory (trajectory.json or TUM/KITTI format)",
    )
    parser.add_argument(
        "ground_truth",
        type=Path,
        help="Path to ground truth trajectory (TUM or KITTI format)",
    )
    parser.add_argument(
        "--ground-truth-fps",
        type=float,
        default=10.0,
        help="FPS for ground truth (for KITTI format, default: 10.0)",
    )
    args = parser.parse_args()

    # Check file existence before loading
    if not args.estimated.exists():
        print(f"Error: Estimated trajectory not found: {args.estimated}", file=sys.stderr)
        sys.exit(1)

    if not args.ground_truth.exists():
        print(f"Error: Ground truth not found: {args.ground_truth}", file=sys.stderr)
        sys.exit(1)

    # Load trajectories
    print(f"Loading estimated trajectory from {args.estimated}")
    try:
        est_interp = load_trajectory(args.estimated)
        print(f"  Loaded {len(est_interp)} poses, duration: {est_interp.duration:.2f}s")
    except Exception as e:
        print(f"Error loading estimated trajectory: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading ground truth from {args.ground_truth}")
    try:
        gt_interp = load_trajectory(args.ground_truth, fps=args.ground_truth_fps)
        print(f"  Loaded {len(gt_interp)} poses, duration: {gt_interp.duration:.2f}s")
    except Exception as e:
        print(f"Error loading ground truth: {e}", file=sys.stderr)
        sys.exit(1)

    # Compute trajectory error
    print("\nComputing trajectory error...")
    error_result = compute_trajectory_error(gt_interp, est_interp)

    print(f"\nAbsolute Trajectory Error (ATE):")
    print(f"  RMSE:   {error_result.ate_rmse:.4f} m")
    print(f"  Mean:   {error_result.ate_mean:.4f} m")
    print(f"  Median: {error_result.ate_median:.4f} m")
    print(f"  Std:    {error_result.ate_std:.4f} m")
    print(f"  Min:    {error_result.ate_min:.4f} m")
    print(f"  Max:    {error_result.ate_max:.4f} m")

    # Generate HTML
    print("\nGenerating HTML visualization...")
    html_content = create_plot_html(gt_interp, est_interp, error_result, scale=1.0)

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html_content)
        html_path = f.name

    print(f"HTML saved to: {html_path}")

    # Open in browser
    print("Opening in browser...")
    webbrowser.open(f"file://{html_path}")

    print("\nDone! The HTML file contains all interactive controls:")
    print("  - Axis remapping dropdowns (remap X/Y/Z axes)")
    print("  - Swap Y/Z button (Y-up to Z-up conversion)")
    print("  - Scale slider (adjust estimated trajectory scale)")
    print("  - X/Y/Z offset sliders (translate estimated trajectory)")
    print("  - Lock offsets checkbox (adjust all offsets together)")
    print("  - Auto-align button (apply optimal Umeyama alignment)")
    print("  - Auto-find Best Axes button (try all 48 axis configurations)")
    print("\nThe visualization updates in real-time as you adjust the controls.")


if __name__ == "__main__":
    main()
