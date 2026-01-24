#include <visual_odometry/motion_estimator.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

namespace visual_odometry {

auto camera_intrinsics::camera_matrix() const -> cv::Mat {
    return (cv::Mat_<double>(3, 3) <<
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0);
}

auto camera_intrinsics::load_from_yaml(std::string_view filepath)
    -> tl::expected<camera_intrinsics, std::string>
{
    cv::FileStorage fs(std::string(filepath), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        return tl::unexpected("Could not open camera file: " + std::string(filepath));
    }

    camera_intrinsics intrinsics;
    fs["fx"] >> intrinsics.fx;
    fs["fy"] >> intrinsics.fy;
    fs["cx"] >> intrinsics.cx;
    fs["cy"] >> intrinsics.cy;

    return intrinsics;
}

auto estimate_motion(std::span<cv::Point2f const> points1,
                     std::span<cv::Point2f const> points2,
                     camera_intrinsics const& intrinsics,
                     motion_estimator_config const& config)
    -> motion_estimate
{
    motion_estimate result;
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

    cv::Mat const K = intrinsics.camera_matrix();
    cv::Mat mask;

    // Compute Essential matrix using RANSAC
    cv::Mat const E = cv::findEssentialMat(
        pts1_mat, pts2_mat, K,
        cv::RANSAC, config.ransac_confidence, config.ransac_threshold, mask);

    if (E.empty()) {
        return result;
    }

    // Recover rotation and translation from Essential matrix
    cv::Mat R_cv, t_cv;
    int const inliers = cv::recoverPose(E, pts1_mat, pts2_mat, K, R_cv, t_cv, mask);

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
