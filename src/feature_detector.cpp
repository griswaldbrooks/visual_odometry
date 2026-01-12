#include <visual_odometry/feature_detector.hpp>
#include <opencv2/imgproc.hpp>

namespace visual_odometry {

FeatureDetector::FeatureDetector(int max_features)
    : orb_(cv::ORB::create(max_features)) {}

void FeatureDetector::detect(const cv::Mat& image,
                              std::vector<cv::KeyPoint>& keypoints,
                              cv::Mat& descriptors) const {
    orb_->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}

std::vector<cv::KeyPoint> FeatureDetector::detectKeypoints(const cv::Mat& image) const {
    std::vector<cv::KeyPoint> keypoints;
    orb_->detect(image, keypoints);
    return keypoints;
}

cv::Mat FeatureDetector::drawKeypoints(const cv::Mat& image,
                                        const std::vector<cv::KeyPoint>& keypoints) {
    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output,
                      cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return output;
}

}  // namespace visual_odometry
