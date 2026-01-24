#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <tl/expected.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <utility>

namespace visual_odometry {

/**
 * @brief Loads sequential images from a directory for visual odometry.
 *
 * Supports PNG and JPEG images. Images are sorted alphabetically by filename.
 */
class image_loader {
public:
    /**
     * @brief Create a new Image Loader.
     * @param image_directory Path to directory containing images.
     * @return image_loader or error message if directory does not exist.
     */
    [[nodiscard]] static auto create(std::string_view image_directory)
        -> tl::expected<image_loader, std::string>;

    /**
     * @brief Load a single image by index.
     * @param index Image index (0-based).
     * @return Grayscale image or error message if index is invalid.
     */
    [[nodiscard]] auto load_image(size_t index) const
        -> tl::expected<cv::Mat, std::string>;

    /**
     * @brief Load a pair of consecutive images.
     * @param index Index of first image.
     * @return Pair of (image[index], image[index+1]) or error.
     */
    [[nodiscard]] auto load_image_pair(size_t index) const
        -> tl::expected<std::pair<cv::Mat, cv::Mat>, std::string>;

    /**
     * @brief Get the next pair of images and advance the index.
     * @return Pair of consecutive images or error.
     */
    [[nodiscard]] auto next_pair()
        -> tl::expected<std::pair<cv::Mat, cv::Mat>, std::string>;

    /**
     * @brief Check if more image pairs are available.
     * @return true if next_pair() can be called.
     */
    [[nodiscard]] auto has_next() const noexcept -> bool;

    /**
     * @brief Reset to the first image.
     */
    auto reset() noexcept -> void;

    /**
     * @brief Get total number of images.
     * @return Number of images in the directory.
     */
    [[nodiscard]] auto size() const noexcept -> size_t;

private:
    // Private constructor - use create() factory
    explicit image_loader(std::string image_directory, std::vector<std::string> image_paths);

    auto load_image_paths() -> tl::expected<void, std::string>;

    std::string image_directory_;
    std::vector<std::string> image_paths_;
    size_t current_index_{0};
};

}  // namespace visual_odometry
