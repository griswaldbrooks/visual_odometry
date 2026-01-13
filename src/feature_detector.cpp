#include <visual_odometry/feature_detector.hpp>
#include <opencv2/imgproc.hpp>

namespace visual_odometry {

FeatureDetector::FeatureDetector(int max_features)
    : orb_(cv::ORB::create(max_features)) {}

auto FeatureDetector::detect(cv::Mat const& image) const -> DetectionResult {
    DetectionResult result;
    orb_->detectAndCompute(image, cv::noArray(), result.keypoints, result.descriptors);
    return result;
}

auto FeatureDetector::detect_keypoints(cv::Mat const& image) const
    -> std::vector<cv::KeyPoint>
{
    std::vector<cv::KeyPoint> keypoints;
    orb_->detect(image, keypoints);
    return keypoints;
}

auto FeatureDetector::draw_keypoints(cv::Mat const& image,
                                      std::vector<cv::KeyPoint> const& keypoints)
    -> cv::Mat
{
    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output,
                      cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return output;
}

}  // namespace visual_odometry
