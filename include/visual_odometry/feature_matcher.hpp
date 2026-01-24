#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <span>
#include <vector>

namespace visual_odometry {

/// Default Lowe's ratio test threshold
constexpr auto default_ratio_threshold = 0.75f;

/**
 * @brief Result of feature matching between two images.
 */
struct match_result {
    std::vector<cv::Point2f> points1;  ///< Matched points in first image
    std::vector<cv::Point2f> points2;  ///< Matched points in second image
    std::vector<cv::DMatch> matches;   ///< Match descriptors
};

/**
 * @brief Configuration for feature matching.
 */
struct feature_matcher_config {
    float ratio_threshold{default_ratio_threshold};
};

// Pure functional API

/**
 * @brief Match features between two sets of descriptors.
 * @param descriptors1 Descriptors from first image.
 * @param descriptors2 Descriptors from second image.
 * @param keypoints1 Keypoints from first image.
 * @param keypoints2 Keypoints from second image.
 * @param config Matching configuration.
 * @return match_result containing matched points and matches.
 */
[[nodiscard]] auto match_features(cv::Mat const& descriptors1,
                                   cv::Mat const& descriptors2,
                                   std::span<cv::KeyPoint const> keypoints1,
                                   std::span<cv::KeyPoint const> keypoints2,
                                   feature_matcher_config const& config = {})
    -> match_result;

/**
 * @brief Draw matches between two images.
 * @param image1 First image.
 * @param keypoints1 Keypoints in first image.
 * @param image2 Second image.
 * @param keypoints2 Keypoints in second image.
 * @param matches Matches to draw.
 * @return Image with matches drawn.
 */
[[nodiscard]] auto draw_feature_matches(cv::Mat const& image1,
                                         std::vector<cv::KeyPoint> const& keypoints1,
                                         cv::Mat const& image2,
                                         std::vector<cv::KeyPoint> const& keypoints2,
                                         std::vector<cv::DMatch> const& matches)
    -> cv::Mat;

}  // namespace visual_odometry
