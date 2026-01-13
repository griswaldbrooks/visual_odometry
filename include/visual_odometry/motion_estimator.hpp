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

/// Default RANSAC reprojection threshold in pixels
constexpr auto default_ransac_threshold = 1.0;

/// Default RANSAC confidence level
constexpr auto default_ransac_confidence = 0.999;

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
 * @brief Result of motion estimation between two frames.
 */
struct MotionEstimate {
    Eigen::Matrix3d rotation;     ///< Rotation matrix (R)
    Eigen::Vector3d translation;  ///< Translation vector (t), unit norm
    int inliers;                  ///< Number of RANSAC inliers
    bool valid;                   ///< Whether estimation succeeded
};

/**
 * @brief Estimates camera motion from matched feature points.
 */
class MotionEstimator {
public:
    /**
     * @brief Construct a new Motion Estimator.
     * @param intrinsics Camera intrinsic parameters.
     * @param ransac_threshold RANSAC reprojection threshold in pixels.
     * @param ransac_confidence RANSAC confidence level (0-1).
     */
    explicit MotionEstimator(CameraIntrinsics const& intrinsics,
                             double ransac_threshold = default_ransac_threshold,
                             double ransac_confidence = default_ransac_confidence);

    /**
     * @brief Estimate motion from matched points.
     * @param points1 Points in first image.
     * @param points2 Corresponding points in second image.
     * @return MotionEstimate containing R, t if successful.
     */
    [[nodiscard]] auto estimate(std::span<cv::Point2f const> points1,
                                 std::span<cv::Point2f const> points2) const
        -> MotionEstimate;

    /**
     * @brief Get the camera intrinsics.
     */
    [[nodiscard]] auto intrinsics() const noexcept -> CameraIntrinsics const& {
        return intrinsics_;
    }

private:
    CameraIntrinsics intrinsics_;
    double ransac_threshold_;
    double ransac_confidence_;
};

}  // namespace visual_odometry
