#include <visual_odometry/motion_estimator.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <stdexcept>

namespace visual_odometry {

cv::Mat CameraIntrinsics::cameraMatrix() const {
    return (cv::Mat_<double>(3, 3) <<
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0);
}

CameraIntrinsics CameraIntrinsics::loadFromYaml(const std::string& filepath) {
    cv::FileStorage fs(filepath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Could not open camera file: " + filepath);
    }

    CameraIntrinsics intrinsics;
    fs["fx"] >> intrinsics.fx;
    fs["fy"] >> intrinsics.fy;
    fs["cx"] >> intrinsics.cx;
    fs["cy"] >> intrinsics.cy;

    return intrinsics;
}

MotionEstimator::MotionEstimator(const CameraIntrinsics& intrinsics,
                                 double ransac_threshold,
                                 double ransac_confidence)
    : intrinsics_(intrinsics),
      ransacThreshold_(ransac_threshold),
      ransacConfidence_(ransac_confidence) {}

MotionEstimate MotionEstimator::estimate(const std::vector<cv::Point2f>& points1,
                                          const std::vector<cv::Point2f>& points2) const {
    MotionEstimate result;
    result.valid = false;
    result.inliers = 0;

    // Need at least 5 points for Essential matrix
    if (points1.size() < 5 || points2.size() < 5) {
        return result;
    }

    cv::Mat K = intrinsics_.cameraMatrix();
    cv::Mat mask;

    // Compute Essential matrix using RANSAC
    cv::Mat E = cv::findEssentialMat(
        points1, points2, K,
        cv::RANSAC, ransacConfidence_, ransacThreshold_, mask);

    if (E.empty()) {
        return result;
    }

    // Recover rotation and translation from Essential matrix
    cv::Mat R_cv, t_cv;
    int inliers = cv::recoverPose(E, points1, points2, K, R_cv, t_cv, mask);

    if (inliers < 10) {
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
