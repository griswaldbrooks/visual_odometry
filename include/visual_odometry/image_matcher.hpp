#pragma once

#include <visual_odometry/feature_detector.hpp>
#include <visual_odometry/feature_matcher.hpp>
#include <opencv2/core.hpp>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

// Forward declaration for OnnxSession (only available when onnxruntime is linked)
namespace visual_odometry {
class OnnxSession;
}

namespace visual_odometry {

/**
 * @brief Match features between two images using ORB detection and matching.
 * @param img1 First grayscale image.
 * @param img2 Second grayscale image.
 * @param detector_config Configuration for feature detection.
 * @param matcher_config Configuration for feature matching.
 * @return MatchResult containing corresponding points.
 */
[[nodiscard]] auto match_images_orb(cv::Mat const& img1,
                                    cv::Mat const& img2,
                                    FeatureDetectorConfig const& detector_config = {},
                                    FeatureMatcherConfig const& matcher_config = {})
    -> MatchResult;

/**
 * @brief Abstract interface for image-to-image feature matching.
 *
 * This provides a unified interface for different matching backends:
 * - ORB (traditional feature detection + descriptor matching)
 * - LightGlue (learned end-to-end matching)
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
    FeatureDetectorConfig detector_config_;
    FeatureMatcherConfig matcher_config_;
};

/**
 * @brief LightGlue learned feature matcher using ONNX Runtime.
 *
 * Uses the DISK+LightGlue fused model for end-to-end learned matching.
 * Requires ONNX Runtime and a pre-exported ONNX model file.
 */
class LightGlueImageMatcher : public ImageMatcher {
public:
    /**
     * @brief Construct LightGlue matcher from an ONNX model file.
     * @param model_path Path to the disk_lightglue_end2end.onnx model file.
     * @throws std::runtime_error if model loading fails.
     */
    explicit LightGlueImageMatcher(
        std::filesystem::path model_path = "models/disk_lightglue_end2end.onnx");

    // Move-only (contains unique_ptr to OnnxSession)
    LightGlueImageMatcher(LightGlueImageMatcher const&) = delete;
    auto operator=(LightGlueImageMatcher const&) -> LightGlueImageMatcher& = delete;
    LightGlueImageMatcher(LightGlueImageMatcher&&) noexcept;
    auto operator=(LightGlueImageMatcher&&) noexcept -> LightGlueImageMatcher&;
    ~LightGlueImageMatcher() override;

    [[nodiscard]] auto match_images(cv::Mat const& img1,
                                    cv::Mat const& img2) const
        -> MatchResult override;

    [[nodiscard]] auto name() const -> std::string_view override {
        return "LightGlue";
    }

private:
    std::filesystem::path model_path_;
    std::unique_ptr<OnnxSession> session_;
};

/**
 * @brief Factory function to create a matcher by name.
 * @param name Matcher name: "orb" (default), "lightglue".
 * @return Unique pointer to the matcher, or nullptr if unknown.
 */
[[nodiscard]] auto create_matcher(std::string_view name)
    -> std::unique_ptr<ImageMatcher>;

}  // namespace visual_odometry
