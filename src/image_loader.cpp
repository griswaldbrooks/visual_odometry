#include <visual_odometry/image_loader.hpp>
#include <filesystem>
#include <algorithm>
#include <stdexcept>

namespace visual_odometry {

ImageLoader::ImageLoader(const std::string& image_directory)
    : imageDirectory_(image_directory), currentIndex_(0) {
    loadImagePaths();
}

void ImageLoader::loadImagePaths() {
    namespace fs = std::filesystem;

    if (!fs::exists(imageDirectory_)) {
        throw std::runtime_error("Image directory does not exist: " + imageDirectory_);
    }

    for (const auto& entry : fs::directory_iterator(imageDirectory_)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                imagePaths_.push_back(entry.path().string());
            }
        }
    }

    std::sort(imagePaths_.begin(), imagePaths_.end());
}

cv::Mat ImageLoader::loadImage(size_t index) const {
    if (index >= imagePaths_.size()) {
        throw std::out_of_range("Image index out of range");
    }
    return cv::imread(imagePaths_[index], cv::IMREAD_GRAYSCALE);
}

std::pair<cv::Mat, cv::Mat> ImageLoader::loadImagePair(size_t index) const {
    return {loadImage(index), loadImage(index + 1)};
}

std::pair<cv::Mat, cv::Mat> ImageLoader::nextPair() {
    auto pair = loadImagePair(currentIndex_);
    currentIndex_++;
    return pair;
}

bool ImageLoader::hasNext() const {
    return currentIndex_ + 1 < imagePaths_.size();
}

void ImageLoader::reset() {
    currentIndex_ = 0;
}

size_t ImageLoader::size() const {
    return imagePaths_.size();
}

}  // namespace visual_odometry
