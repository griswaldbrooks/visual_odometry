#pragma once

#include <visual_odometry/feature_detector.hpp>
#include <visual_odometry/feature_matcher.hpp>
#include <visual_odometry/matcher_concept.hpp>
#include <opencv2/core.hpp>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

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
 * @deprecated Use image_matcher variant with std::visit instead.
 *
 * This class is kept only for Python bindings backward compatibility.
 * C++ code should use the image_matcher variant type which provides
 * zero-overhead polymorphism without virtual dispatch.
 *
 * This provides a unified interface for different matching backends:
 * - ORB (traditional feature detection + descriptor matching)
 * - LightGlue (learned end-to-end matching)
 */
class [[deprecated("Use image_matcher variant instead")]] ImageMatcher {
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
 *
 * Thin wrapper around match_images_orb() pure function.
 * Satisfies the matcher concept without inheritance overhead.
 */
struct orb_matcher {
    /**
     * @brief Construct ORB matcher with default parameters.
     * @param max_features Maximum features to detect (default: 2000).
     * @param ratio_threshold Lowe's ratio test threshold (default: 0.75).
     */
    explicit orb_matcher(int max_features = default_max_features,
                         float ratio_threshold = default_ratio_threshold)
        : detector_config_{.max_features = max_features}
        , matcher_config_{.ratio_threshold = ratio_threshold} {}

    [[nodiscard]] auto match_images(cv::Mat const& img1,
                                    cv::Mat const& img2) const
        -> MatchResult
    {
        return match_images_orb(img1, img2, detector_config_, matcher_config_);
    }

    [[nodiscard]] auto name() const noexcept -> std::string_view {
        return "ORB";
    }

private:
    FeatureDetectorConfig detector_config_;
    FeatureMatcherConfig matcher_config_;
};

/**
 * @brief ORB-based image matcher (detect + match).
 * @deprecated Use orb_matcher instead.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
class OrbImageMatcher : public ImageMatcher {
#pragma GCC diagnostic pop
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
 * Satisfies the matcher concept without inheritance overhead.
 */
struct lightglue_matcher {
    /**
     * @brief Construct LightGlue matcher from an ONNX model file.
     * @param model_path Path to the disk_lightglue_end2end.onnx model file.
     * @throws std::runtime_error if model loading fails.
     */
    explicit lightglue_matcher(
        std::filesystem::path model_path = "models/disk_lightglue_end2end.onnx");

    // Move-only (contains unique_ptr to OnnxSession)
    lightglue_matcher(lightglue_matcher const&) = delete;
    auto operator=(lightglue_matcher const&) -> lightglue_matcher& = delete;
    lightglue_matcher(lightglue_matcher&&) noexcept;
    auto operator=(lightglue_matcher&&) noexcept -> lightglue_matcher&;
    ~lightglue_matcher();

    [[nodiscard]] auto match_images(cv::Mat const& img1,
                                    cv::Mat const& img2) const
        -> MatchResult;

    [[nodiscard]] auto name() const noexcept -> std::string_view {
        return "LightGlue";
    }

private:
    std::filesystem::path model_path_;
    std::unique_ptr<OnnxSession> session_;
};

// Verify matcher concept satisfaction
static_assert(matcher<orb_matcher>, "orb_matcher must satisfy matcher concept");
static_assert(matcher<lightglue_matcher>, "lightglue_matcher must satisfy matcher concept");

/**
 * @brief Variant type holding any image matcher implementation.
 *
 * Replaces the ImageMatcher abstract base class with a std::variant.
 * Provides zero-overhead type-safe polymorphism via std::visit.
 * All alternatives are guaranteed to satisfy the matcher concept.
 */
using image_matcher = std::variant<orb_matcher, lightglue_matcher>;

/**
 * @brief LightGlue learned feature matcher using ONNX Runtime.
 * @deprecated Use lightglue_matcher instead.
 *
 * Uses the DISK+LightGlue fused model for end-to-end learned matching.
 * Requires ONNX Runtime and a pre-exported ONNX model file.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
class LightGlueImageMatcher : public ImageMatcher {
#pragma GCC diagnostic pop
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
 * @brief Factory function to create a matcher by name (variant version).
 * @param name Matcher name: "orb" (default), "lightglue".
 * @return image_matcher variant containing the requested matcher.
 * @throws std::runtime_error if matcher name is unknown.
 */
[[nodiscard]] auto create_image_matcher(std::string_view name)
    -> image_matcher;

/**
 * @brief Factory function to create a matcher by name.
 * @param name Matcher name: "orb" (default), "lightglue".
 * @return Unique pointer to the matcher, or nullptr if unknown.
 * @deprecated Use create_image_matcher() instead which returns image_matcher variant.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
[[nodiscard]] auto create_matcher(std::string_view name)
    -> std::unique_ptr<ImageMatcher>;
#pragma GCC diagnostic pop

}  // namespace visual_odometry
