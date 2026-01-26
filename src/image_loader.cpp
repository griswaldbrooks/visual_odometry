#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <tl/expected.hpp>
#include <visual_odometry/image_loader.hpp>

namespace visual_odometry {

namespace {

/**
 * @brief Parse TUM-format rgb.txt file.
 *
 * Format: lines of "timestamp filename" where lines starting with # are comments.
 *
 * @param rgb_txt_path Path to rgb.txt file.
 * @param base_directory Base directory for resolving relative filenames.
 * @return Map of absolute image path to timestamp, or empty map on error.
 */
auto parse_rgb_txt(std::filesystem::path const& rgb_txt_path,
                   std::filesystem::path const& base_directory)
    -> std::unordered_map<std::string, double> {
    std::unordered_map<std::string, double> timestamp_map;

    std::ifstream file(rgb_txt_path);
    if (!file.is_open()) {
        return timestamp_map;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        double timestamp = 0.0;
        std::string filename;

        if (iss >> timestamp >> filename) {
            // Construct absolute path
            auto const absolute_path = base_directory / filename;
            timestamp_map[absolute_path.string()] = timestamp;
        }
    }

    return timestamp_map;
}

}  // namespace

image_loader::image_loader(std::string image_directory, std::vector<std::string> image_paths,
                           std::vector<double> timestamps)
    : image_directory_(std::move(image_directory)),
      image_paths_(std::move(image_paths)),
      timestamps_(std::move(timestamps)) {}

auto image_loader::create(std::string_view image_directory)
    -> tl::expected<image_loader, std::string> {
    namespace fs = std::filesystem;

    if (!fs::exists(image_directory)) {
        return tl::unexpected("Image directory does not exist: " + std::string(image_directory));
    }

    // Check for rgb.txt in parent directory (TUM format: parent/rgb.txt, parent/rgb/*.png)
    fs::path const img_dir_path(image_directory);
    fs::path const parent_dir = img_dir_path.parent_path();
    fs::path const rgb_txt_path = parent_dir / "rgb.txt";

    std::unordered_map<std::string, double> timestamp_map;
    if (fs::exists(rgb_txt_path)) {
        timestamp_map = parse_rgb_txt(rgb_txt_path, parent_dir);
    }

    std::vector<std::string> image_paths;
    std::vector<double> timestamps;

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

    // Build timestamps vector matching image_paths order
    timestamps.reserve(image_paths.size());
    for (auto const& path : image_paths) {
        auto const it = timestamp_map.find(path);
        if (it != timestamp_map.end()) {
            timestamps.push_back(it->second);
        } else {
            timestamps.push_back(0.0);
        }
    }

    return image_loader(std::string(image_directory), std::move(image_paths),
                        std::move(timestamps));
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

auto image_loader::load_image_with_timestamp(size_t index) const
    -> tl::expected<timestamped_image, std::string> {
    auto const img_result = load_image(index);
    if (!img_result.has_value()) {
        return tl::unexpected(img_result.error());
    }

    return timestamped_image{.image = img_result.value(), .timestamp = get_timestamp(index)};
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

auto image_loader::load_image_pair_with_timestamps(size_t index) const
    -> tl::expected<timestamped_image_pair, std::string> {
    auto const img1 = load_image_with_timestamp(index);
    if (!img1.has_value()) {
        return tl::unexpected(img1.error());
    }

    auto const img2 = load_image_with_timestamp(index + 1);
    if (!img2.has_value()) {
        return tl::unexpected(img2.error());
    }

    return timestamped_image_pair{.first = img1.value(), .second = img2.value()};
}

auto image_loader::next_pair() -> tl::expected<std::pair<cv::Mat, cv::Mat>, std::string> {
    auto pair = load_image_pair(current_index_);
    if (pair.has_value()) {
        current_index_++;
    }
    return pair;
}

auto image_loader::next_pair_with_timestamps()
    -> tl::expected<timestamped_image_pair, std::string> {
    auto pair = load_image_pair_with_timestamps(current_index_);
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

auto image_loader::get_timestamp(size_t index) const noexcept -> double {
    if (index >= timestamps_.size()) {
        return 0.0;
    }
    return timestamps_[index];
}

auto image_loader::has_timestamps() const noexcept -> bool {
    // Check if any timestamp is non-zero
    for (auto const ts : timestamps_) {
        if (ts != 0.0) {
            return true;
        }
    }
    return false;
}

}  // namespace visual_odometry
