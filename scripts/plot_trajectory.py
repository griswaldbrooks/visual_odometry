#!/usr/bin/env python3
"""Plot a single trajectory's X, Y, Z components vs time.

Simple visualization to verify trajectory data before comparison.

Usage:
    python scripts/plot_trajectory.py trajectory.json
    python scripts/plot_trajectory.py data/tum/.../groundtruth.txt
    python scripts/plot_trajectory.py /tmp/circle.txt --fps 30
"""

import argparse
import webbrowser
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from trajectory_utils import load_trajectory


def create_plot_html(timestamps, positions, title: str) -> str:
    """Create HTML with plotly X/Y/Z vs time plots."""

    t = timestamps.tolist()
    x = positions[:, 0].tolist()
    y = positions[:, 1].tolist()
    z = positions[:, 2].tolist()

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #1a1a1a;
            color: #fff;
        }}
        h1 {{ color: #4CAF50; margin-bottom: 5px; }}
        .info {{ color: #888; margin-bottom: 20px; }}
        .plot {{ width: 100%; height: 280px; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="info">
        Poses: {len(timestamps)} |
        Duration: {timestamps[-1] - timestamps[0]:.2f}s |
        X range: [{min(x):.2f}, {max(x):.2f}] |
        Y range: [{min(y):.2f}, {max(y):.2f}] |
        Z range: [{min(z):.2f}, {max(z):.2f}]
    </div>

    <div id="plot_x" class="plot"></div>
    <div id="plot_y" class="plot"></div>
    <div id="plot_z" class="plot"></div>

    <script>
        var layout_base = {{
            paper_bgcolor: '#1a1a1a',
            plot_bgcolor: '#1a1a1a',
            font: {{ color: '#fff' }},
            margin: {{ t: 40, r: 20, b: 50, l: 60 }},
            xaxis: {{ title: 'Time (s)', gridcolor: '#333' }},
            yaxis: {{ gridcolor: '#333' }},
        }};

        var t = {t};
        var x = {x};
        var y = {y};
        var z = {z};

        Plotly.newPlot('plot_x', [
            {{ x: t, y: x, type: 'scatter', mode: 'lines', name: 'X',
               line: {{ color: '#f44336', width: 2 }} }}
        ], {{...layout_base, title: 'X vs Time', yaxis: {{...layout_base.yaxis, title: 'X (m)'}}}});

        Plotly.newPlot('plot_y', [
            {{ x: t, y: y, type: 'scatter', mode: 'lines', name: 'Y',
               line: {{ color: '#4CAF50', width: 2 }} }}
        ], {{...layout_base, title: 'Y vs Time', yaxis: {{...layout_base.yaxis, title: 'Y (m)'}}}});

        Plotly.newPlot('plot_z', [
            {{ x: t, y: z, type: 'scatter', mode: 'lines', name: 'Z',
               line: {{ color: '#2196F3', width: 2 }} }}
        ], {{...layout_base, title: 'Z vs Time', yaxis: {{...layout_base.yaxis, title: 'Z (m)'}}}});
    </script>
</body>
</html>
"""
    return html


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot trajectory X/Y/Z vs time"
    )
    parser.add_argument(
        "trajectory",
        type=Path,
        help="Path to trajectory file (trajectory.json, TUM, or KITTI format)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="FPS for formats without timestamps (default: 10)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser, just print HTML path",
    )
    args = parser.parse_args()

    # Load trajectory
    print(f"Loading trajectory from {args.trajectory}")
    interp = load_trajectory(args.trajectory, fps=args.fps)
    print(f"  Loaded {len(interp)} poses")
    print(f"  Duration: {interp.duration:.2f}s")
    print(f"  Time range: [{interp.start_time:.2f}, {interp.end_time:.2f}]")

    # Create plot
    title = args.trajectory.name
    html = create_plot_html(interp.timestamps, interp.positions, title)

    # Write to temp file and open
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        f.write(html)
        html_path = f.name

    print(f"\nPlot saved to: {html_path}")

    if not args.no_browser:
        print("Opening in browser...")
        webbrowser.open(f"file://{html_path}")


if __name__ == "__main__":
    main()
