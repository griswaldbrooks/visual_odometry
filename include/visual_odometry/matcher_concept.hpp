#pragma once

#include <visual_odometry/feature_matcher.hpp>
#include <opencv2/core.hpp>
#include <concepts>
#include <string_view>

namespace visual_odometry {

/**
 * @brief Concept defining the interface for image matchers.
 *
 * A type satisfies the matcher concept if it provides:
 * - match_images(img1, img2) -> MatchResult
 * - name() -> convertible to string_view
 *
 * This concept is used to constrain types in std::variant and provide
 * compile-time interface checking without virtual dispatch overhead.
 */
template<typename T>
concept matcher = requires(T m, cv::Mat const& img) {
    { m.match_images(img, img) } -> std::same_as<MatchResult>;
    { m.name() } -> std::convertible_to<std::string_view>;
};

}  // namespace visual_odometry
