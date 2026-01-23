#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tl/expected.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace visual_odometry {

/// Minimum inliers required for valid motion estimate
constexpr auto min_motion_inliers = 10;

/// Minimum points required for Essential matrix computation
constexpr auto min_essential_points = 5;

/**
 * @brief Camera intrinsic parameters.
 */
struct CameraIntrinsics {
    double fx;  ///< Focal length x
    double fy;  ///< Focal length y
    double cx;  ///< Principal point x
    double cy;  ///< Principal point y

    /**
     * @brief Get camera matrix as 3x3 cv::Mat.
     */
    [[nodiscard]] auto camera_matrix() const -> cv::Mat;

    /**
     * @brief Load intrinsics from YAML file.
     * @param filepath Path to YAML file.
     * @return CameraIntrinsics or error message.
     */
    [[nodiscard]] static auto load_from_yaml(std::string_view filepath)
        -> tl::expected<CameraIntrinsics, std::string>;
};

/**
 * @brief Configuration parameters for motion estimation.
 */
struct MotionEstimatorConfig {
    double ransac_threshold{1.0};   ///< RANSAC reprojection threshold in pixels
    double ransac_confidence{0.999}; ///< RANSAC confidence level (0-1)
};

/**
 * @brief Result of motion estimation between two frames.
 */
struct MotionEstimate {
    Eigen::Matrix3d rotation;     ///< Rotation matrix (R)
    Eigen::Vector3d translation;  ///< Translation vector (t), unit norm
    int inliers;                  ///< Number of RANSAC inliers
    bool valid;                   ///< Whether estimation succeeded
};

/**
 * @brief Estimate camera motion from matched feature points.
 * @param points1 Points in first image.
 * @param points2 Corresponding points in second image.
 * @param intrinsics Camera intrinsic parameters.
 * @param config Configuration for RANSAC parameters.
 * @return MotionEstimate containing R, t if successful.
 */
[[nodiscard]] auto estimate_motion(std::span<cv::Point2f const> points1,
                                    std::span<cv::Point2f const> points2,
                                    CameraIntrinsics const& intrinsics,
                                    MotionEstimatorConfig const& config = {})
    -> MotionEstimate;

}  // namespace visual_odometry
