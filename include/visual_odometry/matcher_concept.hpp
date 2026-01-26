#pragma once

#include <concepts>
#include <string_view>

#include <opencv2/core.hpp>
#include <visual_odometry/feature_matcher.hpp>

namespace visual_odometry {

/**
 * @brief Concept defining the interface for image matchers.
 *
 * A type satisfies the matcher_like concept if it provides:
 * - match_images(img1, img2) -> match_result
 * - name() -> convertible to string_view
 *
 * This concept is used to constrain types in std::variant and provide
 * compile-time interface checking without virtual dispatch overhead.
 */
template <typename T>
concept matcher_like = requires(T m, cv::Mat const& img) {
    { m.match_images(img, img) } -> std::same_as<match_result>;
    { m.name() } -> std::convertible_to<std::string_view>;
};

}  // namespace visual_odometry
