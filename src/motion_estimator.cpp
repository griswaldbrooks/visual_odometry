#include <visual_odometry/motion_estimator.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

namespace visual_odometry {

auto CameraIntrinsics::camera_matrix() const -> cv::Mat {
    return (cv::Mat_<double>(3, 3) <<
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0);
}

auto CameraIntrinsics::load_from_yaml(std::string_view filepath)
    -> tl::expected<CameraIntrinsics, std::string>
{
    cv::FileStorage fs(std::string(filepath), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return tl::unexpected("Could not open camera file: " + std::string(filepath));
    }

    CameraIntrinsics intrinsics;
    fs["fx"] >> intrinsics.fx;
    fs["fy"] >> intrinsics.fy;
    fs["cx"] >> intrinsics.cx;
    fs["cy"] >> intrinsics.cy;

    return intrinsics;
}

MotionEstimator::MotionEstimator(CameraIntrinsics const& intrinsics,
                                 double ransac_threshold,
                                 double ransac_confidence)
    : intrinsics_(intrinsics),
      ransac_threshold_(ransac_threshold),
      ransac_confidence_(ransac_confidence) {}

auto MotionEstimator::estimate(std::span<cv::Point2f const> points1,
                                std::span<cv::Point2f const> points2) const
    -> MotionEstimate
{
    MotionEstimate result;
    result.valid = false;
    result.inliers = 0;

    // Need at least min_essential_points for Essential matrix
    if (points1.size() < min_essential_points || points2.size() < min_essential_points) {
        return result;
    }

    // Convert spans to cv::Mat for OpenCV functions
    cv::Mat const pts1_mat(static_cast<int>(points1.size()), 1, CV_32FC2,
                           const_cast<cv::Point2f*>(points1.data()));
    cv::Mat const pts2_mat(static_cast<int>(points2.size()), 1, CV_32FC2,
                           const_cast<cv::Point2f*>(points2.data()));

    cv::Mat K = intrinsics_.camera_matrix();
    cv::Mat mask;

    // Compute Essential matrix using RANSAC
    cv::Mat E = cv::findEssentialMat(
        pts1_mat, pts2_mat, K,
        cv::RANSAC, ransac_confidence_, ransac_threshold_, mask);

    if (E.empty()) {
        return result;
    }

    // Recover rotation and translation from Essential matrix
    cv::Mat R_cv, t_cv;
    int inliers = cv::recoverPose(E, pts1_mat, pts2_mat, K, R_cv, t_cv, mask);

    if (inliers < min_motion_inliers) {
        return result;
    }

    // Convert to Eigen
    cv::cv2eigen(R_cv, result.rotation);
    cv::cv2eigen(t_cv, result.translation);

    result.inliers = inliers;
    result.valid = true;

    return result;
}

}  // namespace visual_odometry
