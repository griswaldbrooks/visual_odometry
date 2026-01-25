#!/usr/bin/env python3
"""Plot a single trajectory's X, Y, Z components vs time.

Simple visualization to verify trajectory data before comparison.

Usage:
    python scripts/plot_trajectory.py trajectory.json
    python scripts/plot_trajectory.py data/tum/.../groundtruth.txt
    python scripts/plot_trajectory.py /tmp/circle.txt --fps 30
    python scripts/plot_trajectory.py trajectory.json --images data/tum/.../rgb
"""

import argparse
import webbrowser
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from trajectory_utils import load_trajectory


def get_image_paths_with_timestamps(image_dir: Path) -> list[tuple[float, str]]:
    """Get image paths with their timestamps.

    For TUM format, timestamps are extracted from filenames (e.g., 1305031102.175304.png).
    For other formats, assumes sequential ordering and returns index as timestamp.

    Returns list of (timestamp, absolute_path) tuples, sorted by timestamp.
    """
    image_extensions = {'.png', '.jpg', '.jpeg'}
    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    result = []
    for i, path in enumerate(image_paths):
        # Try to parse TUM-style timestamp from filename
        stem = path.stem
        try:
            timestamp = float(stem)
        except ValueError:
            # Not a timestamp filename, use index
            timestamp = float(i)

        result.append((timestamp, str(path.absolute())))

    return sorted(result, key=lambda x: x[0])


def match_images_to_timestamps(
    trajectory_timestamps: list[float],
    image_data: list[tuple[float, str]],
    max_time_diff: float = 0.1,
) -> list[str | None]:
    """Match trajectory timestamps to nearest images.

    For each trajectory timestamp, finds the image with the closest timestamp.
    Returns None for timestamps outside the image time range (beyond max_time_diff).

    Args:
        trajectory_timestamps: List of trajectory timestamps
        image_data: List of (timestamp, path) tuples
        max_time_diff: Maximum time difference to consider a match (seconds)

    Returns:
        List of image paths (or None) aligned with trajectory timestamps.
    """
    import numpy as np

    if not image_data:
        return [None] * len(trajectory_timestamps)

    image_timestamps = np.array([t for t, _ in image_data])
    image_paths = [p for _, p in image_data]

    matched_paths = []
    for t in trajectory_timestamps:
        # Find closest image timestamp
        diffs = np.abs(image_timestamps - t)
        idx = np.argmin(diffs)

        # Only match if within max_time_diff
        if diffs[idx] <= max_time_diff:
            matched_paths.append(image_paths[idx])
        else:
            matched_paths.append(None)

    return matched_paths


def create_plot_html(timestamps, positions, title: str, image_paths: list[str | None] = None) -> str:
    """Create HTML with plotly X/Y/Z vs time plots and 3D view with synced slider."""

    t = timestamps.tolist()
    x = positions[:, 0].tolist()
    y = positions[:, 1].tolist()
    z = positions[:, 2].tolist()

    has_images = image_paths and len(image_paths) > 0
    # Convert to file:// URLs for local viewing, keep None as null
    if has_images:
        image_urls = [f"file://{p}" if p else None for p in image_paths]
        # Build JSON manually to handle None -> null
        images_json = "[" + ",".join(
            f'"{url}"' if url else "null" for url in image_urls
        ) + "]"
        # Calculate coverage for the indicator bar
        coverage = [1 if p else 0 for p in image_paths]
        coverage_json = str(coverage)
    else:
        images_json = "[]"
        coverage_json = "[]"

    # Adjust layout based on whether we have images
    if has_images:
        container_style = "display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;"
        image_section = """
        <div class="image-panel">
            <h3>Frame Image</h3>
            <div id="image-container">
                <img id="frame-image" src="" alt="Frame image" style="max-width: 100%; border-radius: 4px; display: none;">
                <div id="no-image" style="display: none; padding: 40px; text-align: center; background: #333; border-radius: 4px; color: #888;">
                    No image for this timestamp
                </div>
            </div>
            <div id="image-info" style="color: #888; margin-top: 5px; font-size: 12px;"></div>
        </div>"""
    else:
        container_style = "display: flex; flex-wrap: wrap; gap: 15px;"
        image_section = ""

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
        .info {{ color: #888; margin-bottom: 10px; }}
        .slider-container {{
            background: #2a2a2a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            position: sticky;
            bottom: 10px;
        }}
        .slider-container label {{
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
        }}
        .slider-container input[type="range"] {{
            width: 100%;
            height: 20px;
            cursor: pointer;
        }}
        .slider-info {{
            display: flex;
            justify-content: space-between;
            margin-top: 8px;
            color: #888;
        }}
        .current-pos {{
            color: #ff9800;
            font-family: monospace;
        }}
        .coverage-bar {{
            height: 8px;
            background: #333;
            border-radius: 4px;
            margin-top: 8px;
            overflow: hidden;
            display: flex;
        }}
        .coverage-segment {{
            height: 100%;
        }}
        .coverage-label {{
            font-size: 11px;
            color: #666;
            margin-top: 4px;
        }}
        .container {{ display: flex; flex-wrap: wrap; gap: 15px; }}
        .left {{ flex: 1; min-width: 400px; }}
        .right {{ flex: 1; min-width: 400px; }}
        .plot {{ width: 100%; height: 200px; margin-bottom: 8px; }}
        .plot3d {{ width: 100%; height: 480px; }}
        .image-panel {{ background: #2a2a2a; padding: 15px; border-radius: 8px; }}
        .image-panel h3 {{ margin: 0 0 10px 0; color: #4CAF50; }}
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

    <div class="container" style="{container_style}">
        <div class="left">
            <div id="plot_x" class="plot"></div>
            <div id="plot_y" class="plot"></div>
            <div id="plot_z" class="plot"></div>
        </div>
        <div class="right">
            <div id="plot_3d" class="plot3d"></div>
        </div>
        {image_section}
    </div>

    <div class="slider-container">
        <label>Time Scrubber: <span id="time-display" class="current-pos">0.00s</span></label>
        <input type="range" id="time-slider" min="0" max="{len(timestamps)-1}" value="0" step="1">
        <div class="slider-info">
            <span>Frame: <span id="frame-display">0</span> / {len(timestamps)-1}</span>
            <span class="current-pos">Position: (<span id="pos-display">0, 0, 0</span>)</span>
        </div>
        <div id="coverage-bar" class="coverage-bar"></div>
        <div class="coverage-label"><span style="color:#4CAF50;">■</span> Has image &nbsp; <span style="color:#333;">■</span> No image</div>
    </div>

    <script>
        var t = {t};
        var x = {x};
        var y = {y};
        var z = {z};
        var images = {images_json};
        var coverage = {coverage_json};

        var layout_base = {{
            paper_bgcolor: '#1a1a1a',
            plot_bgcolor: '#1a1a1a',
            font: {{ color: '#fff' }},
            margin: {{ t: 30, r: 20, b: 40, l: 50 }},
            xaxis: {{ title: 'Time (s)', gridcolor: '#333' }},
            yaxis: {{ gridcolor: '#333' }},
            showlegend: false
        }};

        // Create 2D plots with current position marker (trace index 1)
        Plotly.newPlot('plot_x', [
            {{ x: t, y: x, type: 'scatter', mode: 'lines', name: 'X',
               line: {{ color: '#f44336', width: 2 }} }},
            {{ x: [t[0]], y: [x[0]], type: 'scatter', mode: 'markers', name: 'Current',
               marker: {{ size: 12, color: '#ff9800', symbol: 'circle' }} }}
        ], {{...layout_base, title: 'X vs Time', yaxis: {{...layout_base.yaxis, title: 'X (m)'}}}});

        Plotly.newPlot('plot_y', [
            {{ x: t, y: y, type: 'scatter', mode: 'lines', name: 'Y',
               line: {{ color: '#4CAF50', width: 2 }} }},
            {{ x: [t[0]], y: [y[0]], type: 'scatter', mode: 'markers', name: 'Current',
               marker: {{ size: 12, color: '#ff9800', symbol: 'circle' }} }}
        ], {{...layout_base, title: 'Y vs Time', yaxis: {{...layout_base.yaxis, title: 'Y (m)'}}}});

        Plotly.newPlot('plot_z', [
            {{ x: t, y: z, type: 'scatter', mode: 'lines', name: 'Z',
               line: {{ color: '#2196F3', width: 2 }} }},
            {{ x: [t[0]], y: [z[0]], type: 'scatter', mode: 'markers', name: 'Current',
               marker: {{ size: 12, color: '#ff9800', symbol: 'circle' }} }}
        ], {{...layout_base, title: 'Z vs Time', yaxis: {{...layout_base.yaxis, title: 'Z (m)'}}}});

        // 3D trajectory plot with current position marker (trace index 3)
        Plotly.newPlot('plot_3d', [
            {{
                x: x, y: y, z: z,
                type: 'scatter3d',
                mode: 'lines',
                name: 'Trajectory',
                line: {{ color: '#ff9800', width: 3 }}
            }},
            {{
                x: [x[0]], y: [y[0]], z: [z[0]],
                type: 'scatter3d',
                mode: 'markers',
                name: 'Start',
                marker: {{ size: 6, color: '#4CAF50', symbol: 'circle' }}
            }},
            {{
                x: [x[x.length-1]], y: [y[y.length-1]], z: [z[z.length-1]],
                type: 'scatter3d',
                mode: 'markers',
                name: 'End',
                marker: {{ size: 6, color: '#f44336', symbol: 'diamond' }}
            }},
            {{
                x: [x[0]], y: [y[0]], z: [z[0]],
                type: 'scatter3d',
                mode: 'markers',
                name: 'Current',
                marker: {{ size: 10, color: '#ffffff', symbol: 'circle',
                           line: {{ color: '#ff9800', width: 3 }} }}
            }}
        ], {{
            paper_bgcolor: '#1a1a1a',
            font: {{ color: '#fff' }},
            margin: {{ t: 30, r: 20, b: 30, l: 20 }},
            title: '3D Trajectory',
            scene: {{
                xaxis: {{ title: 'X (m)', gridcolor: '#333', backgroundcolor: '#1a1a1a' }},
                yaxis: {{ title: 'Y (m)', gridcolor: '#333', backgroundcolor: '#1a1a1a' }},
                zaxis: {{ title: 'Z (m)', gridcolor: '#333', backgroundcolor: '#1a1a1a' }},
                bgcolor: '#1a1a1a',
                aspectmode: 'data'
            }},
            legend: {{ x: 0.02, y: 0.98 }}
        }});

        // Slider update function
        var slider = document.getElementById('time-slider');
        var timeDisplay = document.getElementById('time-display');
        var frameDisplay = document.getElementById('frame-display');
        var posDisplay = document.getElementById('pos-display');

        var frameImage = document.getElementById('frame-image');
        var noImage = document.getElementById('no-image');
        var imageInfo = document.getElementById('image-info');

        // Build coverage bar
        if (coverage.length > 0) {{
            var coverageBar = document.getElementById('coverage-bar');
            var segments = [];
            var currentVal = coverage[0];
            var count = 1;

            for (var i = 1; i < coverage.length; i++) {{
                if (coverage[i] === currentVal) {{
                    count++;
                }} else {{
                    segments.push({{val: currentVal, count: count}});
                    currentVal = coverage[i];
                    count = 1;
                }}
            }}
            segments.push({{val: currentVal, count: count}});

            var total = coverage.length;
            segments.forEach(function(seg) {{
                var div = document.createElement('div');
                div.className = 'coverage-segment';
                div.style.width = (seg.count / total * 100) + '%';
                div.style.backgroundColor = seg.val ? '#4CAF50' : '#333';
                coverageBar.appendChild(div);
            }});
        }}

        function updatePosition(idx) {{
            var currentT = t[idx];
            var currentX = x[idx];
            var currentY = y[idx];
            var currentZ = z[idx];

            // Update displays
            timeDisplay.textContent = currentT.toFixed(2) + 's';
            frameDisplay.textContent = idx;
            posDisplay.textContent = currentX.toFixed(3) + ', ' + currentY.toFixed(3) + ', ' + currentZ.toFixed(3);

            // Update 2D plot markers
            Plotly.restyle('plot_x', {{ x: [[currentT]], y: [[currentX]] }}, [1]);
            Plotly.restyle('plot_y', {{ x: [[currentT]], y: [[currentY]] }}, [1]);
            Plotly.restyle('plot_z', {{ x: [[currentT]], y: [[currentZ]] }}, [1]);

            // Update 3D plot current marker
            Plotly.restyle('plot_3d', {{ x: [[currentX]], y: [[currentY]], z: [[currentZ]] }}, [3]);

            // Update image if available
            if (images.length > 0 && frameImage && noImage) {{
                var imgUrl = images[idx];
                if (imgUrl) {{
                    frameImage.src = imgUrl;
                    frameImage.style.display = 'block';
                    noImage.style.display = 'none';
                    if (imageInfo) imageInfo.textContent = 'Frame ' + idx;
                }} else {{
                    frameImage.style.display = 'none';
                    noImage.style.display = 'block';
                    if (imageInfo) imageInfo.textContent = 'No image available for this timestamp';
                }}
            }}
        }}

        // Initialize
        updatePosition(0);

        slider.addEventListener('input', function() {{
            updatePosition(parseInt(this.value));
        }});

        // Keyboard controls
        document.addEventListener('keydown', function(e) {{
            var idx = parseInt(slider.value);
            if (e.key === 'ArrowRight' || e.key === 'l') {{
                slider.value = Math.min(idx + 1, t.length - 1);
                updatePosition(parseInt(slider.value));
            }} else if (e.key === 'ArrowLeft' || e.key === 'h') {{
                slider.value = Math.max(idx - 1, 0);
                updatePosition(parseInt(slider.value));
            }} else if (e.key === 'Home' || e.key === 'g') {{
                slider.value = 0;
                updatePosition(0);
            }} else if (e.key === 'End' || e.key === 'G') {{
                slider.value = t.length - 1;
                updatePosition(t.length - 1);
            }}
        }});
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
        "--images",
        type=Path,
        default=None,
        help="Path to image directory (shows corresponding frame)",
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

    # Load and match images if provided
    image_paths = None
    if args.images and args.images.exists():
        print(f"Loading images from {args.images}")
        image_data = get_image_paths_with_timestamps(args.images)
        print(f"  Found {len(image_data)} images")

        # Get image time range
        img_start = image_data[0][0]
        img_end = image_data[-1][0]
        print(f"  Image time range: [{img_start:.3f}, {img_end:.3f}]")

        # Match images to trajectory timestamps (returns None for out-of-range)
        image_paths = match_images_to_timestamps(interp.timestamps.tolist(), image_data)
        n_matched = sum(1 for p in image_paths if p is not None)
        n_missing = len(image_paths) - n_matched
        print(f"  Matched: {n_matched} poses with images, {n_missing} without")

    # Make timestamps relative to first sample
    timestamps = interp.timestamps - interp.timestamps[0]

    # Create plot
    title = str(args.trajectory)
    html = create_plot_html(timestamps, interp.positions, title, image_paths)

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
