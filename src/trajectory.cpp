#include <visual_odometry/trajectory.hpp>
#include <visual_odometry/motion_estimator.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <tl/expected.hpp>
#include <Eigen/Dense>

namespace visual_odometry {

auto pose::identity() noexcept -> pose {
    return pose{
        .rotation = Eigen::Matrix3d::Identity(),  // NOLINT(misc-include-cleaner)
        .translation = Eigen::Vector3d::Zero()    // NOLINT(misc-include-cleaner)
    };
}

auto pose::compose(motion_estimate const& relative) const noexcept -> pose {
    // T_world_new = T_world_current * T_current_new
    // R_new = R_current * R_relative
    // t_new = R_current * t_relative + t_current
    pose result;
    result.rotation = rotation * relative.rotation;
    result.translation = rotation * relative.translation + translation;
    return result;
}

Trajectory::Trajectory() {
    poses_.push_back(pose::identity());
}

auto Trajectory::add_motion(motion_estimate const& motion) -> bool {
    if (!motion.valid) {
        return false;
    }

    auto const new_pose = poses_.back().compose(motion);
    poses_.push_back(new_pose);
    return true;
}

auto Trajectory::poses() const noexcept -> std::vector<pose> const& {
    return poses_;
}

auto Trajectory::current_pose() const noexcept -> pose const& {
    return poses_.back();
}

auto Trajectory::size() const noexcept -> size_t {
    return poses_.size();
}

auto Trajectory::empty() const noexcept -> bool {
    return poses_.size() <= 1;  // Only origin
}

auto Trajectory::reset() noexcept -> void {
    poses_.clear();
    poses_.push_back(pose::identity());
}

auto Trajectory::to_json() const -> std::string {
    std::ostringstream oss;
    oss << std::setprecision(10);
    oss << "{\n  \"poses\": [\n";

    for (size_t i = 0; i < poses_.size(); ++i) {
        auto const& pose = poses_[i];
        oss << "    {\n";

        // Rotation matrix as 3x3 array
        oss << "      \"rotation\": [\n";
        for (int row = 0; row < 3; ++row) {
            oss << "        [";
            for (int col = 0; col < 3; ++col) {
                oss << pose.rotation(row, col);
                if (col < 2) {
                    oss << ", ";
                }
            }
            oss << "]";
            if (row < 2) {
                oss << ",";
            }
            oss << "\n";
        }
        oss << "      ],\n";

        // Translation as array
        oss << "      \"translation\": [";
        oss << pose.translation(0) << ", ";
        oss << pose.translation(1) << ", ";
        oss << pose.translation(2) << "]\n";

        oss << "    }";
        if (i < poses_.size() - 1) {
            oss << ",";
        }
        oss << "\n";
    }

    oss << "  ]\n}\n";
    return oss.str();
}

auto Trajectory::save_to_json(std::filesystem::path const& filepath) const
    -> tl::expected<void, std::string>
{
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return tl::unexpected("Failed to open file for writing: " + filepath.string());
    }

    file << to_json();

    if (!file.good()) {
        return tl::unexpected("Error writing to file: " + filepath.string());
    }

    return {};
}

}  // namespace visual_odometry
