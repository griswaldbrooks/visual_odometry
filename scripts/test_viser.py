#!/usr/bin/env python3
"""Test script to verify viser installation works."""

import numpy as np
import viser


def main():
    # Create a viser server
    server = viser.ViserServer()
    print(f"Viser server started at: http://localhost:{server.port}")

    # Add a simple coordinate frame at origin
    server.scene.add_frame("world", wxyz=(1, 0, 0, 0), position=(0, 0, 0))

    # Add a sample camera frustum
    server.scene.add_camera_frustum(
        "camera_0",
        fov=60.0,
        aspect=1241 / 376,  # KITTI aspect ratio
        scale=0.3,
        wxyz=(1, 0, 0, 0),
        position=(0, 0, 0),
        color=(0, 255, 0),
    )

    # Add a second camera to show motion
    server.scene.add_camera_frustum(
        "camera_1",
        fov=60.0,
        aspect=1241 / 376,
        scale=0.3,
        wxyz=(1, 0, 0, 0),
        position=(0, 0, 1),  # 1 meter forward
        color=(255, 0, 0),
    )

    # Add a trajectory line
    points = np.array([[0, 0, 0], [0, 0, 1]])
    server.scene.add_spline_catmull_rom("trajectory", points, color=(255, 255, 0))

    print("Open the URL above in your browser to see the visualization.")
    print("Press Ctrl+C to exit.")

    # Keep server running
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
