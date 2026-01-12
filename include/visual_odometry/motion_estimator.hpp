#pragma once

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <optional>

namespace visual_odometry {

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
    cv::Mat cameraMatrix() const;

    /**
     * @brief Load intrinsics from YAML file.
     */
    static CameraIntrinsics loadFromYaml(const std::string& filepath);
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
    explicit MotionEstimator(const CameraIntrinsics& intrinsics,
                             double ransac_threshold = 1.0,
                             double ransac_confidence = 0.999);

    /**
     * @brief Estimate motion from matched points.
     * @param points1 Points in first image.
     * @param points2 Corresponding points in second image.
     * @return MotionEstimate containing R, t if successful.
     */
    MotionEstimate estimate(const std::vector<cv::Point2f>& points1,
                            const std::vector<cv::Point2f>& points2) const;

    /**
     * @brief Get the camera intrinsics.
     */
    const CameraIntrinsics& intrinsics() const { return intrinsics_; }

private:
    CameraIntrinsics intrinsics_;
    double ransacThreshold_;
    double ransacConfidence_;
};

}  // namespace visual_odometry
