#include <visual_odometry/feature_detector.hpp>
#include <opencv2/imgproc.hpp>

namespace visual_odometry {

// Pure functional implementations

[[nodiscard]] auto detect_features(cv::Mat const& image,
                                    feature_detector_config const& config)
    -> detection_result
{
    auto const orb = cv::ORB::create(config.max_features);
    detection_result result;
    orb->detectAndCompute(image, cv::noArray(), result.keypoints, result.descriptors);
    return result;
}

[[nodiscard]] auto detect_keypoints_only(cv::Mat const& image,
                                         feature_detector_config const& config)
    -> std::vector<cv::KeyPoint>
{
    auto const orb = cv::ORB::create(config.max_features);
    std::vector<cv::KeyPoint> keypoints;
    orb->detect(image, keypoints);
    return keypoints;
}

[[nodiscard]] auto draw_keypoints(cv::Mat const& image,
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
