#include <visual_odometry/feature_matcher.hpp>

namespace visual_odometry {

FeatureMatcher::FeatureMatcher(float ratio_threshold)
    : matcher_(cv::BFMatcher::create(cv::NORM_HAMMING, false)),
      ratioThreshold_(ratio_threshold) {}

MatchResult FeatureMatcher::match(const cv::Mat& descriptors1,
                                   const cv::Mat& descriptors2,
                                   const std::vector<cv::KeyPoint>& keypoints1,
                                   const std::vector<cv::KeyPoint>& keypoints2) const {
    MatchResult result;

    if (descriptors1.empty() || descriptors2.empty()) {
        return result;
    }

    // KNN match with k=2 for ratio test
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher_->knnMatch(descriptors1, descriptors2, knnMatches, 2);

    // Apply Lowe's ratio test
    for (const auto& match : knnMatches) {
        if (match.size() >= 2 && match[0].distance < ratioThreshold_ * match[1].distance) {
            result.matches.push_back(match[0]);
            result.points1.push_back(keypoints1[static_cast<size_t>(match[0].queryIdx)].pt);
            result.points2.push_back(keypoints2[static_cast<size_t>(match[0].trainIdx)].pt);
        }
    }

    return result;
}

cv::Mat FeatureMatcher::drawMatches(const cv::Mat& image1,
                                     const std::vector<cv::KeyPoint>& keypoints1,
                                     const cv::Mat& image2,
                                     const std::vector<cv::KeyPoint>& keypoints2,
                                     const std::vector<cv::DMatch>& matches) {
    cv::Mat output;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, output,
                    cv::Scalar(0, 255, 0),   // Match color (green)
                    cv::Scalar(255, 0, 0),   // Single point color (blue)
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return output;
}

}  // namespace visual_odometry
