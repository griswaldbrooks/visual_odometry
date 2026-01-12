#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace visual_odometry {

/**
 * @brief Result of feature matching between two images.
 */
struct MatchResult {
    std::vector<cv::Point2f> points1;  ///< Matched points in first image
    std::vector<cv::Point2f> points2;  ///< Matched points in second image
    std::vector<cv::DMatch> matches;   ///< Match descriptors
};

/**
 * @brief Matches ORB features between two images.
 */
class FeatureMatcher {
public:
    /**
     * @brief Construct a new Feature Matcher.
     * @param ratio_threshold Lowe's ratio test threshold (default 0.75).
     */
    explicit FeatureMatcher(float ratio_threshold = 0.75f);

    /**
     * @brief Match features between two sets of descriptors.
     * @param descriptors1 Descriptors from first image.
     * @param descriptors2 Descriptors from second image.
     * @param keypoints1 Keypoints from first image.
     * @param keypoints2 Keypoints from second image.
     * @return MatchResult containing matched points and matches.
     */
    MatchResult match(const cv::Mat& descriptors1,
                      const cv::Mat& descriptors2,
                      const std::vector<cv::KeyPoint>& keypoints1,
                      const std::vector<cv::KeyPoint>& keypoints2) const;

    /**
     * @brief Draw matches between two images.
     * @param image1 First image.
     * @param keypoints1 Keypoints in first image.
     * @param image2 Second image.
     * @param keypoints2 Keypoints in second image.
     * @param matches Matches to draw.
     * @return Image with matches drawn.
     */
    static cv::Mat drawMatches(const cv::Mat& image1,
                               const std::vector<cv::KeyPoint>& keypoints1,
                               const cv::Mat& image2,
                               const std::vector<cv::KeyPoint>& keypoints2,
                               const std::vector<cv::DMatch>& matches);

private:
    cv::Ptr<cv::BFMatcher> matcher_;
    float ratioThreshold_;
};

}  // namespace visual_odometry
