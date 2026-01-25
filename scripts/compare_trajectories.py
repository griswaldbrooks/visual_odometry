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


def add_trajectory_to_scene(
    server: viser.ViserServer,
    interp: TrajectoryInterpolator,
    name: str,
    color: tuple[int, int, int],
    frustum_interval: int = 10,
) -> None:
    """Add a trajectory to the viser scene.

    Args:
        server: Viser server instance
        interp: Trajectory interpolator
        name: Name prefix for scene elements
        color: RGB color tuple
        frustum_interval: Add camera frustum every N frames
    """
    positions = interp.positions

    # Add trajectory line
    if len(positions) > 1:
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

        server.scene.add_camera_frustum(
            f"{name}/camera_{i:04d}",
            fov=60.0,
            aspect=4 / 3,
            scale=0.15,
            wxyz=wxyz,
            position=tuple(pose.position),
            color=color,
        )


def create_plot_html(
    gt_interp: TrajectoryInterpolator,
    est_interp: TrajectoryInterpolator,
    error_result,
) -> str:
    """Create HTML with plotly time-series plots.

    Returns HTML string with embedded plotly charts.
    """
    # Get data
    gt_t = gt_interp.timestamps
    gt_pos = gt_interp.positions
    est_t = est_interp.timestamps
    est_pos = est_interp.positions
    err_t = error_result.timestamps
    err_pos = error_result.position_errors
    err_total = error_result.total_errors

    # Create plotly figure data as JSON
    html = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 10px; background: #1a1a1a; color: #fff; }
        .plot { width: 100%; height: 200px; margin-bottom: 10px; }
        .stats { background: #2a2a2a; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
        h3 { margin: 5px 0; color: #4CAF50; }
    </style>
</head>
<body>
    <div class="stats">
        <h3>Absolute Trajectory Error (ATE)</h3>
        <table>
            <tr><td>RMSE:</td><td><b>""" + f"{error_result.ate_rmse:.4f}" + """ m</b></td></tr>
            <tr><td>Mean:</td><td>""" + f"{error_result.ate_mean:.4f}" + """ m</td></tr>
            <tr><td>Median:</td><td>""" + f"{error_result.ate_median:.4f}" + """ m</td></tr>
            <tr><td>Std:</td><td>""" + f"{error_result.ate_std:.4f}" + """ m</td></tr>
            <tr><td>Min:</td><td>""" + f"{error_result.ate_min:.4f}" + """ m</td></tr>
            <tr><td>Max:</td><td>""" + f"{error_result.ate_max:.4f}" + """ m</td></tr>
        </table>
    </div>

    <div id="plot_x" class="plot"></div>
    <div id="plot_y" class="plot"></div>
    <div id="plot_z" class="plot"></div>
    <div id="plot_error" class="plot"></div>

    <script>
        var layout = {
            paper_bgcolor: '#1a1a1a',
            plot_bgcolor: '#1a1a1a',
            font: { color: '#fff' },
            margin: { t: 30, r: 20, b: 40, l: 50 },
            xaxis: { title: 'Time (s)', gridcolor: '#333' },
            yaxis: { gridcolor: '#333' },
            legend: { x: 0, y: 1 }
        };

        // X vs Time
        Plotly.newPlot('plot_x', [
            { x: """ + str(gt_t.tolist()) + """, y: """ + str(gt_pos[:, 0].tolist()) + """,
              name: 'GT X', line: { color: '#4CAF50' } },
            { x: """ + str(est_t.tolist()) + """, y: """ + str(est_pos[:, 0].tolist()) + """,
              name: 'Est X', line: { color: '#f44336' } }
        ], {...layout, yaxis: {...layout.yaxis, title: 'X (m)'}});

        // Y vs Time
        Plotly.newPlot('plot_y', [
            { x: """ + str(gt_t.tolist()) + """, y: """ + str(gt_pos[:, 1].tolist()) + """,
              name: 'GT Y', line: { color: '#4CAF50' } },
            { x: """ + str(est_t.tolist()) + """, y: """ + str(est_pos[:, 1].tolist()) + """,
              name: 'Est Y', line: { color: '#f44336' } }
        ], {...layout, yaxis: {...layout.yaxis, title: 'Y (m)'}});

        // Z vs Time
        Plotly.newPlot('plot_z', [
            { x: """ + str(gt_t.tolist()) + """, y: """ + str(gt_pos[:, 2].tolist()) + """,
              name: 'GT Z', line: { color: '#4CAF50' } },
            { x: """ + str(est_t.tolist()) + """, y: """ + str(est_pos[:, 2].tolist()) + """,
              name: 'Est Z', line: { color: '#f44336' } }
        ], {...layout, yaxis: {...layout.yaxis, title: 'Z (m)'}});

        // Error vs Time
        Plotly.newPlot('plot_error', [
            { x: """ + str(err_t.tolist()) + """, y: """ + str(err_total.tolist()) + """,
              name: 'Total Error', line: { color: '#ff9800' }, fill: 'tozeroy' },
            { x: """ + str(err_t.tolist()) + """, y: """ + str(np.abs(err_pos[:, 0]).tolist()) + """,
              name: 'X Error', line: { color: '#2196F3', dash: 'dot' } },
            { x: """ + str(err_t.tolist()) + """, y: """ + str(np.abs(err_pos[:, 1]).tolist()) + """,
              name: 'Y Error', line: { color: '#9C27B0', dash: 'dot' } },
            { x: """ + str(err_t.tolist()) + """, y: """ + str(np.abs(err_pos[:, 2]).tolist()) + """,
              name: 'Z Error', line: { color: '#00BCD4', dash: 'dot' } }
        ], {...layout, yaxis: {...layout.yaxis, title: 'Error (m)'}});
    </script>
</body>
</html>
"""
    return html


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

    # Compute time-matched error
    print("\nComputing time-matched trajectory error...")
    error_result = compute_trajectory_error(gt_interp, est_interp)

    print(f"\nAbsolute Trajectory Error (ATE):")
    print(f"  RMSE:   {error_result.ate_rmse:.4f} m")
    print(f"  Mean:   {error_result.ate_mean:.4f} m")
    print(f"  Median: {error_result.ate_median:.4f} m")
    print(f"  Std:    {error_result.ate_std:.4f} m")
    print(f"  Min:    {error_result.ate_min:.4f} m")
    print(f"  Max:    {error_result.ate_max:.4f} m")

    # Start viser server
    server = viser.ViserServer()
    print(f"\nViser server started at: http://localhost:{server.get_port()}")

    # Add world frame
    server.scene.add_frame("world", wxyz=(1, 0, 0, 0), position=(0, 0, 0), axes_length=1.0)

    # Add trajectories to 3D scene
    print("Adding trajectories to 3D scene...")
    add_trajectory_to_scene(server, gt_interp, "gt", (0, 255, 0))  # Green
    add_trajectory_to_scene(server, est_interp, "est", (255, 0, 0))  # Red

    # Create and serve plots HTML
    print("Generating time-series plots...")
    plots_html = create_plot_html(gt_interp, est_interp, error_result)

    # Write plots to temp file and serve via panel
    import tempfile
    import webbrowser

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(plots_html)
        plots_path = f.name

    print(f"\nTime-series plots saved to: {plots_path}")
    print(f"Opening plots in browser...")
    webbrowser.open(f"file://{plots_path}")

    print("\n3D visualization: http://localhost:8080")
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
