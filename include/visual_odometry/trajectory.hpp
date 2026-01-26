#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <visual_odometry/motion_estimator.hpp>

namespace visual_odometry {

/**
 * @brief Represents an absolute camera pose in the world frame.
 */
struct pose {
    Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};
    Eigen::Vector3d translation{Eigen::Vector3d::Zero()};

    /**
     * @brief Create identity pose (origin).
     */
    [[nodiscard]] static auto identity() noexcept -> pose;

    /**
     * @brief Compose this pose with a relative transform.
     * @param relative The relative motion (R, t) from this pose.
     * @return New absolute pose after applying the relative transform.
     */
    [[nodiscard]] auto compose(motion_estimate const& relative) const noexcept -> pose;
};

/**
 * @brief Accumulates camera poses to build a trajectory.
 */
class Trajectory {
public:
    /**
     * @brief Construct empty trajectory starting at origin.
     */
    Trajectory();

    /**
     * @brief Add a relative motion estimate to the trajectory.
     * @param motion Relative motion from current to next frame.
     * @return true if motion was valid and added, false otherwise.
     */
    auto add_motion(motion_estimate const& motion) -> bool;

    /**
     * @brief Get all poses in the trajectory.
     * @return Vector of absolute poses, starting from origin.
     */
    [[nodiscard]] auto poses() const noexcept -> std::vector<pose> const&;

    /**
     * @brief Get the current (latest) pose.
     */
    [[nodiscard]] auto current_pose() const noexcept -> pose const&;

    /**
     * @brief Get the number of poses in the trajectory.
     */
    [[nodiscard]] auto size() const noexcept -> size_t;

    /**
     * @brief Check if trajectory is empty (only has origin).
     */
    [[nodiscard]] auto empty() const noexcept -> bool;

    /**
     * @brief Reset trajectory to origin.
     */
    auto reset() noexcept -> void;

    /**
     * @brief Save trajectory to JSON file.
     * @param filepath Output file path.
     * @return true if saved successfully, false on error.
     */
    [[nodiscard]] auto save_to_json(std::filesystem::path const& filepath) const
        -> tl::expected<void, std::string>;

    /**
     * @brief Convert trajectory to JSON string.
     * @return JSON string representation of the trajectory.
     */
    [[nodiscard]] auto to_json() const -> std::string;

private:
    std::vector<pose> poses_;
};

}  // namespace visual_odometry
