#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace visual_odometry {

/// Default maximum number of features to detect
constexpr auto default_max_features = 2000;

/**
 * @brief Result of feature detection containing keypoints and descriptors.
 */
struct DetectionResult {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

/**
 * @brief Detects ORB features in images.
 */
class FeatureDetector {
public:
    /**
     * @brief Construct a new Feature Detector.
     * @param max_features Maximum number of features to detect.
     */
    explicit FeatureDetector(int max_features = default_max_features);

    /**
     * @brief Detect keypoints and compute descriptors.
     * @param image Grayscale input image.
     * @return DetectionResult containing keypoints and descriptors.
     */
    [[nodiscard]] auto detect(cv::Mat const& image) const -> DetectionResult;

    /**
     * @brief Detect keypoints only (no descriptors).
     * @param image Grayscale input image.
     * @return Vector of detected keypoints.
     */
    [[nodiscard]] auto detect_keypoints(cv::Mat const& image) const
        -> std::vector<cv::KeyPoint>;

    /**
     * @brief Draw keypoints on an image.
     * @param image Input image.
     * @param keypoints Keypoints to draw.
     * @return Image with keypoints drawn.
     */
    [[nodiscard]] static auto draw_keypoints(cv::Mat const& image,
                                              std::vector<cv::KeyPoint> const& keypoints)
        -> cv::Mat;

private:
    cv::Ptr<cv::ORB> orb_;
};

}  // namespace visual_odometry
