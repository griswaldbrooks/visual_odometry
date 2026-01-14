#pragma once

#include <visual_odometry/feature_detector.hpp>
#include <visual_odometry/feature_matcher.hpp>
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <string_view>

namespace visual_odometry {

/**
 * @brief Abstract interface for image-to-image feature matching.
 *
 * This provides a unified interface for different matching backends:
 * - ORB (traditional feature detection + descriptor matching)
 * - Learned matchers like MatchAnything (end-to-end matching)
 */
class ImageMatcher {
public:
    virtual ~ImageMatcher() = default;

    /**
     * @brief Match features between two images.
     * @param img1 First grayscale image.
     * @param img2 Second grayscale image.
     * @return MatchResult containing corresponding points.
     */
    [[nodiscard]] virtual auto match_images(cv::Mat const& img1,
                                            cv::Mat const& img2) const
        -> MatchResult = 0;

    /**
     * @brief Get the name of this matcher backend.
     */
    [[nodiscard]] virtual auto name() const -> std::string_view = 0;
};

/**
 * @brief ORB-based image matcher (detect + match).
 */
class OrbImageMatcher : public ImageMatcher {
public:
    /**
     * @brief Construct ORB matcher with default parameters.
     * @param max_features Maximum features to detect (default: 2000).
     * @param ratio_threshold Lowe's ratio test threshold (default: 0.75).
     */
    explicit OrbImageMatcher(int max_features = default_max_features,
                             float ratio_threshold = default_ratio_threshold);

    [[nodiscard]] auto match_images(cv::Mat const& img1,
                                    cv::Mat const& img2) const
        -> MatchResult override;

    [[nodiscard]] auto name() const -> std::string_view override {
        return "ORB";
    }

private:
    FeatureDetector detector_;
    FeatureMatcher matcher_;
};

/**
 * @brief Factory function to create a matcher by name.
 * @param name Matcher name: "orb" (default), "matchanything" (future).
 * @return Unique pointer to the matcher, or nullptr if unknown.
 */
[[nodiscard]] auto create_matcher(std::string_view name)
    -> std::unique_ptr<ImageMatcher>;

}  // namespace visual_odometry
