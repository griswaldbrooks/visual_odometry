"""Visual odometry library - Python bindings.

This module provides Python bindings for the C++ visual odometry library,
enabling feature detection, matching, motion estimation, and trajectory
accumulation for camera pose estimation from sequential images.

Example:
    >>> import visual_odometry as vo
    >>>
    >>> # Load images and camera
    >>> loader = vo.ImageLoader.create("path/to/images")
    >>> intrinsics = vo.CameraIntrinsics.load_from_yaml("camera.yaml")
    >>>
    >>> # Set up pipeline
    >>> matcher = vo.create_matcher("orb")
    >>> estimator = vo.MotionEstimator(intrinsics)
    >>> trajectory = vo.Trajectory()
    >>>
    >>> # Process images
    >>> for img1, img2 in loader:
    ...     result = matcher.match_images(img1, img2)
    ...     motion = estimator.estimate(result.points1, result.points2)
    ...     trajectory.add_motion(motion)
    >>>
    >>> # Save results
    >>> trajectory.save_to_json("trajectory.json")
"""

from _visual_odometry_impl import (
    __version__,
    # Image loading
    ImageLoader,
    # Feature matching
    MatchResult,
    ImageMatcher,
    OrbImageMatcher,
    create_matcher,
    # Motion estimation
    CameraIntrinsics,
    MotionEstimate,
    MotionEstimator,
    # Trajectory
    Pose,
    Trajectory,
)

__all__ = [
    "__version__",
    "ImageLoader",
    "MatchResult",
    "ImageMatcher",
    "OrbImageMatcher",
    "create_matcher",
    "CameraIntrinsics",
    "MotionEstimate",
    "MotionEstimator",
    "Pose",
    "Trajectory",
]
