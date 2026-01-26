#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <tl/expected.hpp>
#include <visual_odometry/image_loader.hpp>

namespace visual_odometry {

image_loader::image_loader(std::string image_directory, std::vector<std::string> image_paths)
    : image_directory_(std::move(image_directory)), image_paths_(std::move(image_paths)) {}

auto image_loader::create(std::string_view image_directory)
    -> tl::expected<image_loader, std::string> {
    namespace fs = std::filesystem;

    if (!fs::exists(image_directory)) {
        return tl::unexpected("Image directory does not exist: " + std::string(image_directory));
    }

    std::vector<std::string> image_paths;

    for (auto const& entry : fs::directory_iterator(image_directory)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            std::ranges::transform(ext, ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                image_paths.push_back(entry.path().string());
            }
        }
    }

    std::sort(image_paths.begin(), image_paths.end());  // NOLINT(modernize-use-ranges)

    return image_loader(std::string(image_directory), std::move(image_paths));
}

auto image_loader::load_image(size_t index) const -> tl::expected<cv::Mat, std::string> {
    if (index >= image_paths_.size()) {
        return tl::unexpected("Image index out of range: " + std::to_string(index) +
                              " >= " + std::to_string(image_paths_.size()));
    }

    auto image = cv::imread(image_paths_[index], cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        return tl::unexpected("Failed to load image: " + image_paths_[index]);
    }

    return image;
}

auto image_loader::load_image_pair(size_t index) const
    -> tl::expected<std::pair<cv::Mat, cv::Mat>, std::string> {
    auto const img1 = load_image(index);
    if (!img1.has_value()) {
        return tl::unexpected(img1.error());
    }

    auto const img2 = load_image(index + 1);
    if (!img2.has_value()) {
        return tl::unexpected(img2.error());
    }

    return std::make_pair(img1.value(), img2.value());
}

auto image_loader::next_pair() -> tl::expected<std::pair<cv::Mat, cv::Mat>, std::string> {
    auto pair = load_image_pair(current_index_);
    if (pair.has_value()) {
        current_index_++;
    }
    return pair;
}

auto image_loader::has_next() const noexcept -> bool {
    return current_index_ + 1 < image_paths_.size();
}

auto image_loader::reset() noexcept -> void {
    current_index_ = 0;
}

auto image_loader::size() const noexcept -> size_t {
    return image_paths_.size();
}

}  // namespace visual_odometry
