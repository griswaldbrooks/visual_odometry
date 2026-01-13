#include <visual_odometry/image_loader.hpp>
#include <filesystem>
#include <algorithm>

namespace visual_odometry {

ImageLoader::ImageLoader(std::string image_directory, std::vector<std::string> image_paths)
    : image_directory_(std::move(image_directory)),
      image_paths_(std::move(image_paths)),
      current_index_(0) {}

auto ImageLoader::create(std::string_view image_directory)
    -> tl::expected<ImageLoader, std::string>
{
    namespace fs = std::filesystem;

    if (!fs::exists(image_directory)) {
        return tl::unexpected("Image directory does not exist: " + std::string(image_directory));
    }

    std::vector<std::string> image_paths;

    for (auto const& entry : fs::directory_iterator(image_directory)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                image_paths.push_back(entry.path().string());
            }
        }
    }

    std::sort(image_paths.begin(), image_paths.end());

    return ImageLoader(std::string(image_directory), std::move(image_paths));
}

auto ImageLoader::load_image(size_t index) const
    -> tl::expected<cv::Mat, std::string>
{
    if (index >= image_paths_.size()) {
        return tl::unexpected("Image index out of range: " + std::to_string(index)
            + " >= " + std::to_string(image_paths_.size()));
    }

    auto const image = cv::imread(image_paths_[index], cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        return tl::unexpected("Failed to load image: " + image_paths_[index]);
    }

    return image;
}

auto ImageLoader::load_image_pair(size_t index) const
    -> tl::expected<std::pair<cv::Mat, cv::Mat>, std::string>
{
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

auto ImageLoader::next_pair()
    -> tl::expected<std::pair<cv::Mat, cv::Mat>, std::string>
{
    auto pair = load_image_pair(current_index_);
    if (pair.has_value()) {
        current_index_++;
    }
    return pair;
}

auto ImageLoader::has_next() const noexcept -> bool {
    return current_index_ + 1 < image_paths_.size();
}

auto ImageLoader::reset() noexcept -> void {
    current_index_ = 0;
}

auto ImageLoader::size() const noexcept -> size_t {
    return image_paths_.size();
}

}  // namespace visual_odometry
