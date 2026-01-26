#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <tl/expected.hpp>

namespace visual_odometry {

/**
 * @brief Image with associated timestamp.
 */
struct timestamped_image {
    cv::Mat image;
    double timestamp{0.0};
};

/**
 * @brief Pair of timestamped images.
 */
struct timestamped_image_pair {
    timestamped_image first;
    timestamped_image second;
};

/**
 * @brief Loads sequential images from a directory for visual odometry.
 *
 * Supports PNG and JPEG images. Images are sorted alphabetically by filename.
 * If a TUM-format rgb.txt file is present, timestamps are parsed from it.
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
    [[nodiscard]] auto load_image(size_t index) const -> tl::expected<cv::Mat, std::string>;

    /**
     * @brief Load a single image with its timestamp.
     * @param index Image index (0-based).
     * @return Timestamped image or error message if index is invalid.
     */
    [[nodiscard]] auto load_image_with_timestamp(size_t index) const
        -> tl::expected<timestamped_image, std::string>;

    /**
     * @brief Load a pair of consecutive images.
     * @param index Index of first image.
     * @return Pair of (image[index], image[index+1]) or error.
     */
    [[nodiscard]] auto load_image_pair(size_t index) const
        -> tl::expected<std::pair<cv::Mat, cv::Mat>, std::string>;

    /**
     * @brief Load a pair of consecutive images with timestamps.
     * @param index Index of first image.
     * @return Pair of timestamped images or error.
     */
    [[nodiscard]] auto load_image_pair_with_timestamps(size_t index) const
        -> tl::expected<timestamped_image_pair, std::string>;

    /**
     * @brief Get the next pair of images and advance the index.
     * @return Pair of consecutive images or error.
     */
    [[nodiscard]] auto next_pair() -> tl::expected<std::pair<cv::Mat, cv::Mat>, std::string>;

    /**
     * @brief Get the next pair of images with timestamps and advance the index.
     * @return Pair of timestamped images or error.
     */
    [[nodiscard]] auto next_pair_with_timestamps()
        -> tl::expected<timestamped_image_pair, std::string>;

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

    /**
     * @brief Get timestamp for a specific image.
     * @param index Image index (0-based).
     * @return Timestamp in seconds since epoch, or 0.0 if unavailable.
     */
    [[nodiscard]] auto get_timestamp(size_t index) const noexcept -> double;

    /**
     * @brief Check if timestamps are available.
     * @return true if rgb.txt was parsed successfully.
     */
    [[nodiscard]] auto has_timestamps() const noexcept -> bool;

private:
    // Private constructor - use create() factory
    explicit image_loader(std::string image_directory, std::vector<std::string> image_paths,
                          std::vector<double> timestamps);

    auto load_image_paths() -> tl::expected<void, std::string>;

    std::string image_directory_;
    std::vector<std::string> image_paths_;
    std::vector<double> timestamps_;
    size_t current_index_{0};
};

}  // namespace visual_odometry
