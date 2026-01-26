#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

#include <opencv2/core.hpp>
#include <visual_odometry/feature_detector.hpp>
#include <visual_odometry/feature_matcher.hpp>
#include <visual_odometry/matcher_concept.hpp>

// Forward declaration for onnx_session (only available when onnxruntime is linked)
namespace visual_odometry {
class onnx_session;
}

namespace visual_odometry {

/**
 * @brief Match features between two images using ORB detection and matching.
 * @param img1 First grayscale image.
 * @param img2 Second grayscale image.
 * @param detector_config Configuration for feature detection.
 * @param matcher_config Configuration for feature matching.
 * @return match_result containing corresponding points.
 */
[[nodiscard]] auto match_images_orb(cv::Mat const& img1, cv::Mat const& img2,
                                    feature_detector_config const& detector_config = {},
                                    feature_matcher_config const& matcher_config = {})
    -> match_result;

/**
 * @brief ORB-based image matcher (detect + match).
 *
 * Thin wrapper around match_images_orb() pure function.
 * Satisfies the matcher_like concept without inheritance overhead.
 */
struct orb_matcher {
    /**
     * @brief Construct ORB matcher with default parameters.
     * @param max_features Maximum features to detect (default: 2000).
     * @param ratio_threshold Lowe's ratio test threshold (default: 0.75).
     */
    explicit orb_matcher(int max_features = default_max_features,
                         float ratio_threshold = default_ratio_threshold)
        : detector_config_{.max_features = max_features},
          matcher_config_{.ratio_threshold = ratio_threshold} {}

    [[nodiscard]] auto match_images(cv::Mat const& img1, cv::Mat const& img2) const
        -> match_result {
        return match_images_orb(img1, img2, detector_config_, matcher_config_);
    }

    [[nodiscard]] auto name() const noexcept -> std::string_view { return "ORB"; }

private:
    feature_detector_config detector_config_;
    feature_matcher_config matcher_config_;
};

/**
 * @brief LightGlue learned feature matcher using ONNX Runtime.
 *
 * Uses the DISK+LightGlue fused model for end-to-end learned matching.
 * Requires ONNX Runtime and a pre-exported ONNX model file.
 * Satisfies the matcher_like concept without inheritance overhead.
 */
struct lightglue_matcher {
    /**
     * @brief Construct LightGlue matcher from an ONNX model file.
     * @param model_path Path to the disk_lightglue_end2end.onnx model file.
     * @throws std::runtime_error if model loading fails.
     */
    explicit lightglue_matcher(
        std::filesystem::path model_path = "models/disk_lightglue_end2end.onnx");

    // Move-only (contains unique_ptr to onnx_session)
    lightglue_matcher(lightglue_matcher const&) = delete;
    auto operator=(lightglue_matcher const&) -> lightglue_matcher& = delete;
    lightglue_matcher(lightglue_matcher&&) noexcept;
    auto operator=(lightglue_matcher&&) noexcept -> lightglue_matcher&;
    ~lightglue_matcher();

    [[nodiscard]] auto match_images(cv::Mat const& img1, cv::Mat const& img2) const -> match_result;

    [[nodiscard]] auto name() const noexcept -> std::string_view { return "LightGlue"; }

private:
    std::filesystem::path model_path_;
    std::unique_ptr<onnx_session> session_;
};

// Verify matcher_like concept satisfaction
static_assert(matcher_like<orb_matcher>, "orb_matcher must satisfy matcher_like concept");
static_assert(matcher_like<lightglue_matcher>,
              "lightglue_matcher must satisfy matcher_like concept");

/**
 * @brief Variant type holding any image matcher implementation.
 *
 * Replaces the ImageMatcher abstract base class with a std::variant.
 * Provides zero-overhead type-safe polymorphism via std::visit.
 * All alternatives are guaranteed to satisfy the matcher_like concept.
 */
using image_matcher = std::variant<orb_matcher, lightglue_matcher>;

/**
 * @brief Factory function to create a matcher by name (variant version).
 * @param name Matcher name: "orb" (default), "lightglue".
 * @return image_matcher variant containing the requested matcher.
 * @throws std::runtime_error if matcher name is unknown.
 */
[[nodiscard]] auto create_image_matcher(std::string_view name) -> image_matcher;

}  // namespace visual_odometry
