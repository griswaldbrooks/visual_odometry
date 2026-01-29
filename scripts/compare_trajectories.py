#!/usr/bin/env python3
"""Interactive 3D trajectory viewer using viser.

Displays:
- 3D trajectory visualization with camera frustums
- Real-time ATE metrics
- Scale adjustment slider
- X/Y/Z offset sliders
- Axis remapping controls
- Auto-align button (Umeyama algorithm)

Note: For time-series plots (X/Y/Z vs Time, Error vs Time), use plot_trajectory_error.py

Usage:
    python scripts/compare_trajectories.py trajectory.json data/tum/.../groundtruth.txt
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
    offset: np.ndarray = None,
    frustum_scale: float = 0.02,
) -> None:
    """Add a trajectory to the viser scene.

    Args:
        server: Viser server instance
        interp: Trajectory interpolator
        name: Name prefix for scene elements
        color: RGB color tuple
        frustum_interval: Add camera frustum every N frames
        scale: Scale factor to apply to positions
        offset: Translation offset to apply (shape: (3,))
        frustum_scale: Size of camera frustums
    """
    if offset is None:
        offset = np.array([0.0, 0.0, 0.0])

    positions = interp.positions * scale + offset

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
        transformed_position = pose.position * scale + offset

        server.scene.add_camera_frustum(
            f"{name}/camera_{i:04d}",
            fov=60.0,
            aspect=1241 / 376,  # KITTI camera aspect ratio
            scale=frustum_scale,
            wxyz=wxyz,
            position=tuple(transformed_position),
            color=color,
        )


def compute_scaled_error(
    gt_interp: TrajectoryInterpolator,
    est_interp: TrajectoryInterpolator,
    scale: float,
    offset: np.ndarray = None,
):
    """Compute trajectory error with scaled and offset estimated positions.

    Args:
        gt_interp: Ground truth trajectory interpolator
        est_interp: Estimated trajectory interpolator
        scale: Scale factor to apply to estimated positions
        offset: Translation offset to apply (shape: (3,))

    Returns:
        TrajectoryErrorResult with errors computed on scaled and offset trajectory
    """
    if offset is None:
        offset = np.array([0.0, 0.0, 0.0])

    # Create a scaled and offset version of the estimated interpolator
    transformed_positions = est_interp.positions * scale + offset
    transformed_interp = TrajectoryInterpolator(
        est_interp.timestamps,
        transformed_positions,
        est_interp.quaternions,
    )
    return compute_trajectory_error(gt_interp, transformed_interp)


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
    est_interp = load_trajectory(args.estimated)
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

    # Add frustum scale slider
    frustum_scale_slider = server.gui.add_slider(
        "Frustum Size",
        min=0.001,
        max=0.2,
        step=0.001,
        initial_value=0.02,
    )

    # Add offset sliders
    offset_x_slider = server.gui.add_slider(
        "Offset X",
        min=-100.0,
        max=100.0,
        step=0.01,
        initial_value=0.0,
    )

    offset_y_slider = server.gui.add_slider(
        "Offset Y",
        min=-100.0,
        max=100.0,
        step=0.01,
        initial_value=0.0,
    )

    offset_z_slider = server.gui.add_slider(
        "Offset Z",
        min=-100.0,
        max=100.0,
        step=0.01,
        initial_value=0.0,
    )

    # Add axis swap dropdowns
    axis_options = ["X", "-X", "Y", "-Y", "Z", "-Z"]
    axis_x_dropdown = server.gui.add_dropdown(
        "Axis X maps to",
        axis_options,
        initial_value="X",
    )

    axis_y_dropdown = server.gui.add_dropdown(
        "Axis Y maps to",
        axis_options,
        initial_value="Y",
    )

    axis_z_dropdown = server.gui.add_dropdown(
        "Axis Z maps to",
        axis_options,
        initial_value="Z",
    )

    # Add preset buttons for common axis swaps
    swap_yz_button = server.gui.add_button("Swap Y/Z (Y-up to Z-up)")
    reset_axes_button = server.gui.add_button("Reset Axes")

    # Add auto-align button
    auto_align_button = server.gui.add_button("Auto-align (Umeyama)")
    auto_find_axes_button = server.gui.add_button("Auto-find Best Axes")

    def parse_axis_mapping(value: str) -> tuple[int, bool]:
        """Parse axis dropdown value to (axis_index, negate).

        Returns:
            (axis_index, negate) where axis_index is 0=X, 1=Y, 2=Z
        """
        axis_map = {'X': 0, 'Y': 1, 'Z': 2}
        if value.startswith('-'):
            return axis_map[value[1]], True
        return axis_map[value], False

    def apply_axis_remapping(positions: np.ndarray) -> np.ndarray:
        """Apply current axis remapping to positions array."""
        x_idx, x_neg = parse_axis_mapping(axis_x_dropdown.value)
        y_idx, y_neg = parse_axis_mapping(axis_y_dropdown.value)
        z_idx, z_neg = parse_axis_mapping(axis_z_dropdown.value)

        remapped = np.zeros_like(positions)
        remapped[:, 0] = -positions[:, x_idx] if x_neg else positions[:, x_idx]
        remapped[:, 1] = -positions[:, y_idx] if y_neg else positions[:, y_idx]
        remapped[:, 2] = -positions[:, z_idx] if z_neg else positions[:, z_idx]
        return remapped

    def apply_axis_config(positions: np.ndarray, x_map: str, y_map: str, z_map: str) -> np.ndarray:
        """Apply specific axis mapping to positions array."""
        axis_map = {'X': 0, 'Y': 1, 'Z': 2}

        def parse(value):
            if value.startswith('-'):
                return axis_map[value[1]], True
            return axis_map[value], False

        x_idx, x_neg = parse(x_map)
        y_idx, y_neg = parse(y_map)
        z_idx, z_neg = parse(z_map)

        remapped = np.zeros_like(positions)
        remapped[:, 0] = -positions[:, x_idx] if x_neg else positions[:, x_idx]
        remapped[:, 1] = -positions[:, y_idx] if y_neg else positions[:, y_idx]
        remapped[:, 2] = -positions[:, z_idx] if z_neg else positions[:, z_idx]
        return remapped

    def find_best_axis_configuration():
        """Try all valid axis permutations and find the one with lowest RMSE."""
        import itertools

        axes = ['X', 'Y', 'Z']
        signs = ['', '-']

        best_rmse = float('inf')
        best_config = ('X', 'Y', 'Z')
        best_scale = 1.0
        best_offset = np.array([0.0, 0.0, 0.0])

        gt_at_est, _ = gt_interp.interpolate_batch(est_interp.timestamps)

        # Try all 6 permutations × 8 sign combinations = 48 configurations
        for perm in itertools.permutations(axes):
            for sx, sy, sz in itertools.product(signs, signs, signs):
                x_map = sx + perm[0]
                y_map = sy + perm[1]
                z_map = sz + perm[2]

                # Apply this axis configuration
                remapped = apply_axis_config(est_interp.positions, x_map, y_map, z_map)

                # Compute Umeyama alignment
                scale, rotation, translation = compute_umeyama_alignment(gt_at_est, remapped)

                # Compute error with this alignment
                transformed = remapped * scale + translation
                errors = np.linalg.norm(gt_at_est - transformed, axis=1)
                rmse = np.sqrt(np.mean(errors ** 2))

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_config = (x_map or 'X', y_map or 'Y', z_map or 'Z')
                    best_scale = scale
                    best_offset = translation

        return best_config, best_scale, best_offset, best_rmse

    # Add a text display for current error metrics
    error_text = server.gui.add_text(
        "ATE Metrics",
        initial_value="Computing...",
        disabled=True,
    )

    def update_visualization(scale: float, offset_x: float, offset_y: float, offset_z: float, redraw_gt: bool = False) -> None:
        """Update the visualization with new scale and offset parameters."""
        frustum_size = frustum_scale_slider.value

        # Remove old estimated trajectory
        remove_trajectory_from_scene(server, est_interp, "est")

        # Optionally redraw ground truth (for frustum size changes)
        if redraw_gt:
            remove_trajectory_from_scene(server, gt_interp, "gt")
            add_trajectory_to_scene(server, gt_interp, "gt", (0, 255, 0), frustum_scale=frustum_size)

        # Create offset array
        offset = np.array([offset_x, offset_y, offset_z])

        # Apply axis remapping to estimated positions
        remapped_positions = apply_axis_remapping(est_interp.positions)

        # Create a temporary interpolator with remapped positions
        remapped_est_interp = TrajectoryInterpolator(
            est_interp.timestamps,
            remapped_positions,
            est_interp.quaternions,
        )

        # Add scaled and offset estimated trajectory (with remapped positions)
        add_trajectory_to_scene(server, remapped_est_interp, "est", (255, 0, 0), scale=scale, offset=offset, frustum_scale=frustum_size)

        # Compute scaled and offset error (with remapped positions)
        error_result = compute_scaled_error(gt_interp, remapped_est_interp, scale, offset=offset)

        # Update error text display
        axis_info = f"[{axis_x_dropdown.value}, {axis_y_dropdown.value}, {axis_z_dropdown.value}]"
        error_text.value = (
            f"RMSE: {error_result.ate_rmse:.4f} m | "
            f"Mean: {error_result.ate_mean:.4f} m | "
            f"Axes: {axis_info}"
        )

        print(f"  Scale: {scale:.3f}, Offset: [{offset_x:.2f}, {offset_y:.2f}, {offset_z:.2f}], Axes: {axis_info} | ATE RMSE: {error_result.ate_rmse:.4f} m")

        return error_result

    # Register callback for scale changes
    @scale_slider.on_update
    def on_scale_change(event: viser.GuiEvent) -> None:
        update_visualization(
            scale_slider.value,
            offset_x_slider.value,
            offset_y_slider.value,
            offset_z_slider.value,
        )

    # Register callbacks for offset sliders
    @offset_x_slider.on_update
    def on_offset_x_change(event: viser.GuiEvent) -> None:
        update_visualization(
            scale_slider.value,
            offset_x_slider.value,
            offset_y_slider.value,
            offset_z_slider.value,
        )

    @offset_y_slider.on_update
    def on_offset_y_change(event: viser.GuiEvent) -> None:
        update_visualization(
            scale_slider.value,
            offset_x_slider.value,
            offset_y_slider.value,
            offset_z_slider.value,
        )

    @offset_z_slider.on_update
    def on_offset_z_change(event: viser.GuiEvent) -> None:
        update_visualization(
            scale_slider.value,
            offset_x_slider.value,
            offset_y_slider.value,
            offset_z_slider.value,
        )

    # Register callback for frustum scale slider
    @frustum_scale_slider.on_update
    def on_frustum_scale_change(event: viser.GuiEvent) -> None:
        update_visualization(
            scale_slider.value,
            offset_x_slider.value,
            offset_y_slider.value,
            offset_z_slider.value,
            redraw_gt=True,  # Need to redraw GT to update its frustum size
        )

    # Register callbacks for axis dropdowns
    @axis_x_dropdown.on_update
    def on_axis_x_change(event: viser.GuiEvent) -> None:
        update_visualization(
            scale_slider.value,
            offset_x_slider.value,
            offset_y_slider.value,
            offset_z_slider.value,
        )

    @axis_y_dropdown.on_update
    def on_axis_y_change(event: viser.GuiEvent) -> None:
        update_visualization(
            scale_slider.value,
            offset_x_slider.value,
            offset_y_slider.value,
            offset_z_slider.value,
        )

    @axis_z_dropdown.on_update
    def on_axis_z_change(event: viser.GuiEvent) -> None:
        update_visualization(
            scale_slider.value,
            offset_x_slider.value,
            offset_y_slider.value,
            offset_z_slider.value,
        )

    # Register callback for Swap Y/Z button
    @swap_yz_button.on_click
    def on_swap_yz(event: viser.GuiEvent) -> None:
        axis_x_dropdown.value = "X"
        axis_y_dropdown.value = "Z"
        axis_z_dropdown.value = "-Y"
        update_visualization(
            scale_slider.value,
            offset_x_slider.value,
            offset_y_slider.value,
            offset_z_slider.value,
        )

    # Register callback for Reset Axes button
    @reset_axes_button.on_click
    def on_reset_axes(event: viser.GuiEvent) -> None:
        axis_x_dropdown.value = "X"
        axis_y_dropdown.value = "Y"
        axis_z_dropdown.value = "Z"
        update_visualization(
            scale_slider.value,
            offset_x_slider.value,
            offset_y_slider.value,
            offset_z_slider.value,
        )

    # Register callback for auto-align button
    @auto_align_button.on_click
    def on_auto_align(event: viser.GuiEvent) -> None:
        # Apply axis remapping to estimated positions first
        remapped_positions = apply_axis_remapping(est_interp.positions)

        # Compute Umeyama alignment with remapped positions
        gt_at_est, _ = gt_interp.interpolate_batch(est_interp.timestamps)
        optimal_scale, optimal_rotation, optimal_translation = compute_umeyama_alignment(
            gt_at_est, remapped_positions
        )

        # Apply optimal values to GUI
        scale_slider.value = optimal_scale
        offset_x_slider.value = optimal_translation[0]
        offset_y_slider.value = optimal_translation[1]
        offset_z_slider.value = optimal_translation[2]

        # Update visualization
        update_visualization(
            optimal_scale,
            optimal_translation[0],
            optimal_translation[1],
            optimal_translation[2],
        )

        print(f"Auto-aligned with axes [{axis_x_dropdown.value}, {axis_y_dropdown.value}, {axis_z_dropdown.value}]: scale={optimal_scale:.6f}, offset={optimal_translation}")

    # Register callback for auto-find best axes button
    @auto_find_axes_button.on_click
    def on_auto_find_axes(event: viser.GuiEvent) -> None:
        print("Searching for best axis configuration (48 combinations)...")
        best_config, best_scale, best_offset, best_rmse = find_best_axis_configuration()

        # Apply best axis configuration to dropdowns
        axis_x_dropdown.value = best_config[0]
        axis_y_dropdown.value = best_config[1]
        axis_z_dropdown.value = best_config[2]

        # Apply optimal scale and offset
        scale_slider.value = best_scale
        offset_x_slider.value = best_offset[0]
        offset_y_slider.value = best_offset[1]
        offset_z_slider.value = best_offset[2]

        # Update visualization
        update_visualization(
            best_scale,
            best_offset[0],
            best_offset[1],
            best_offset[2],
        )

        print(f"Best axes: [{best_config[0]}, {best_config[1]}, {best_config[2]}], scale={best_scale:.6f}, RMSE={best_rmse:.4f}m")

    # Add trajectories to 3D scene
    print("Adding trajectories to 3D scene...")
    add_trajectory_to_scene(server, gt_interp, "gt", (0, 255, 0), frustum_scale=frustum_scale_slider.value)  # Green

    # Initial visualization with scale=1.0, offset=0
    print("\nComputing initial trajectory error...")
    error_result = update_visualization(1.0, 0.0, 0.0, 0.0)

    print(f"\nAbsolute Trajectory Error (ATE) at scale=1.0, offset=[0, 0, 0]:")
    print(f"  RMSE:   {error_result.ate_rmse:.4f} m")
    print(f"  Mean:   {error_result.ate_mean:.4f} m")
    print(f"  Median: {error_result.ate_median:.4f} m")
    print(f"  Std:    {error_result.ate_std:.4f} m")
    print(f"  Min:    {error_result.ate_min:.4f} m")
    print(f"  Max:    {error_result.ate_max:.4f} m")

    print(f"\nViser 3D viewer: http://localhost:{server.get_port()}")
    print("  Green = Ground Truth")
    print("  Red = Estimated (adjustable)")
    print("\nGUI Controls:")
    print("  - 'Estimated Scale' slider: adjust trajectory scale")
    print("  - 'Offset X/Y/Z' sliders: translate estimated trajectory")
    print("  - 'Frustum Size' slider: adjust camera frustum visualization size")
    print("  - 'Auto-align' button: apply optimal Umeyama alignment")
    print("  - 'Auto-find Best Axes' button: try all 48 axis configurations")
    print("  - Axis dropdowns: remap X/Y/Z axes of estimated trajectory")
    print("  - 'Swap Y/Z' button: Y-up to Z-up conversion")
    print("  - 'Reset Axes' button: restore default axis mapping")
    print("\nFor time-series plots (X/Y/Z vs Time, Error vs Time):")
    print("  Run: python scripts/plot_trajectory_error.py <estimated> <ground_truth>")
    print("\nPress Ctrl+C to exit.")

    import time
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
