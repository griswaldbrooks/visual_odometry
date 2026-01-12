#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>
#include <utility>

namespace visual_odometry {

/**
 * @brief Loads sequential images from a directory for visual odometry.
 *
 * Supports PNG and JPEG images. Images are sorted alphabetically by filename.
 */
class ImageLoader {
public:
    /**
     * @brief Construct a new Image Loader.
     * @param image_directory Path to directory containing images.
     * @throws std::runtime_error if directory does not exist.
     */
    explicit ImageLoader(const std::string& image_directory);

    /**
     * @brief Load a single image by index.
     * @param index Image index (0-based).
     * @return Grayscale image.
     * @throws std::out_of_range if index is invalid.
     */
    cv::Mat loadImage(size_t index) const;

    /**
     * @brief Load a pair of consecutive images.
     * @param index Index of first image.
     * @return Pair of (image[index], image[index+1]).
     */
    std::pair<cv::Mat, cv::Mat> loadImagePair(size_t index) const;

    /**
     * @brief Get the next pair of images and advance the index.
     * @return Pair of consecutive images.
     */
    std::pair<cv::Mat, cv::Mat> nextPair();

    /**
     * @brief Check if more image pairs are available.
     * @return true if nextPair() can be called.
     */
    bool hasNext() const;

    /**
     * @brief Reset to the first image.
     */
    void reset();

    /**
     * @brief Get total number of images.
     * @return Number of images in the directory.
     */
    size_t size() const;

private:
    void loadImagePaths();

    std::string imageDirectory_;
    std::vector<std::string> imagePaths_;
    size_t currentIndex_;
};

}  // namespace visual_odometry
