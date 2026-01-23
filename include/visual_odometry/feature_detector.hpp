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
 * @brief Configuration for feature detection.
 */
struct FeatureDetectorConfig {
    int max_features{default_max_features};
};

// Pure functional API

/**
 * @brief Detect keypoints and compute descriptors.
 * @param image Grayscale input image.
 * @param config Detection configuration.
 * @return DetectionResult containing keypoints and descriptors.
 */
[[nodiscard]] auto detect_features(cv::Mat const& image,
                                    FeatureDetectorConfig const& config = {})
    -> DetectionResult;

/**
 * @brief Detect keypoints only (no descriptors).
 * @param image Grayscale input image.
 * @param config Detection configuration.
 * @return Vector of detected keypoints.
 */
[[nodiscard]] auto detect_keypoints_only(cv::Mat const& image,
                                         FeatureDetectorConfig const& config = {})
    -> std::vector<cv::KeyPoint>;

/**
 * @brief Draw keypoints on an image.
 * @param image Input image.
 * @param keypoints Keypoints to draw.
 * @return Image with keypoints drawn.
 */
[[nodiscard]] auto draw_keypoints(cv::Mat const& image,
                                   std::vector<cv::KeyPoint> const& keypoints)
    -> cv::Mat;

}  // namespace visual_odometry
