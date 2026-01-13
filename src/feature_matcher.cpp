#include <visual_odometry/feature_matcher.hpp>

namespace visual_odometry {

FeatureMatcher::FeatureMatcher(float ratio_threshold)
    : matcher_(cv::BFMatcher::create(cv::NORM_HAMMING, false)),
      ratio_threshold_(ratio_threshold) {}

auto FeatureMatcher::match(cv::Mat const& descriptors1,
                            cv::Mat const& descriptors2,
                            std::span<cv::KeyPoint const> keypoints1,
                            std::span<cv::KeyPoint const> keypoints2) const
    -> MatchResult
{
    MatchResult result;

    if (descriptors1.empty() || descriptors2.empty()) {
        return result;
    }

    // KNN match with k=2 for ratio test
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    // Apply Lowe's ratio test
    for (auto const& match : knn_matches) {
        if (match.size() >= 2 && match[0].distance < ratio_threshold_ * match[1].distance) {
            result.matches.push_back(match[0]);
            result.points1.push_back(keypoints1[static_cast<size_t>(match[0].queryIdx)].pt);
            result.points2.push_back(keypoints2[static_cast<size_t>(match[0].trainIdx)].pt);
        }
    }

    return result;
}

auto FeatureMatcher::draw_matches(cv::Mat const& image1,
                                   std::vector<cv::KeyPoint> const& keypoints1,
                                   cv::Mat const& image2,
                                   std::vector<cv::KeyPoint> const& keypoints2,
                                   std::vector<cv::DMatch> const& matches)
    -> cv::Mat
{
    cv::Mat output;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, output,
                    cv::Scalar(0, 255, 0),   // Match color (green)
                    cv::Scalar(255, 0, 0),   // Single point color (blue)
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return output;
}

}  // namespace visual_odometry
