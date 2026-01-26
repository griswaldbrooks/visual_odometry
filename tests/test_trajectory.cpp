#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numbers>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <visual_odometry/trajectory.hpp>

class trajectory_test : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a valid forward motion estimate
        forward_motion_.rotation = Eigen::Matrix3d::Identity();
        forward_motion_.translation = Eigen::Vector3d(0, 0, 1);
        forward_motion_.inliers = 100;
        forward_motion_.valid = true;

        // Create a rotation motion (90 degrees around Y)
        double const angle = std::numbers::pi / 2.0;
        rotation_motion_.rotation =
            Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()).toRotationMatrix();
        rotation_motion_.translation = Eigen::Vector3d(0, 0, 1);
        rotation_motion_.inliers = 100;
        rotation_motion_.valid = true;

        // Create an invalid motion
        invalid_motion_.valid = false;
    }

    visual_odometry::motion_estimate forward_motion_;
    visual_odometry::motion_estimate rotation_motion_;
    visual_odometry::motion_estimate invalid_motion_;
};

TEST_F(trajectory_test, StartsAtOrigin) {
    // GIVEN a new trajectory
    visual_odometry::Trajectory const trajectory;

    // THEN it should have one pose at origin
    EXPECT_EQ(trajectory.size(), 1);
    EXPECT_TRUE(trajectory.empty());

    auto const& pose = trajectory.current_pose();
    EXPECT_TRUE(pose.rotation.isIdentity());
    EXPECT_TRUE(pose.translation.isZero());
}

TEST_F(trajectory_test, AccumulatesForwardMotion) {
    // GIVEN a trajectory
    visual_odometry::Trajectory trajectory;

    // WHEN adding forward motion
    bool const added = trajectory.add_motion(forward_motion_);

    // THEN motion should be added
    EXPECT_TRUE(added);
    EXPECT_EQ(trajectory.size(), 2);
    EXPECT_FALSE(trajectory.empty());

    // AND current pose should be translated forward
    auto const& pose = trajectory.current_pose();
    EXPECT_TRUE(pose.rotation.isIdentity());
    EXPECT_NEAR(pose.translation(2), 1.0, 1e-9);
}

TEST_F(trajectory_test, AccumulatesMultipleMotions) {
    // GIVEN a trajectory
    visual_odometry::Trajectory trajectory;

    // WHEN adding multiple forward motions
    for (int i = 0; i < 5; ++i) {
        trajectory.add_motion(forward_motion_);
    }

    // THEN trajectory should have 6 poses (origin + 5)
    EXPECT_EQ(trajectory.size(), 6);

    // AND final position should be at z=5
    auto const& pose = trajectory.current_pose();
    EXPECT_NEAR(pose.translation(2), 5.0, 1e-9);
}

TEST_F(trajectory_test, ChainsRotationAndTranslation) {
    // GIVEN a trajectory
    visual_odometry::Trajectory trajectory;

    // WHEN adding a 90-degree rotation and then forward motion
    trajectory.add_motion(rotation_motion_);
    trajectory.add_motion(forward_motion_);

    // THEN final position should reflect the rotated frame
    auto const& pose = trajectory.current_pose();

    // After rotation, "forward" (0,0,1) in camera frame becomes (1,0,0) in world
    // So second translation adds to x, not z
    EXPECT_NEAR(pose.translation(0), 1.0, 1e-6);
    EXPECT_NEAR(pose.translation(2), 1.0, 1e-6);  // From first motion
}

TEST_F(trajectory_test, RejectsInvalidMotion) {
    // GIVEN a trajectory
    visual_odometry::Trajectory trajectory;

    // WHEN adding invalid motion
    bool const added = trajectory.add_motion(invalid_motion_);

    // THEN motion should be rejected
    EXPECT_FALSE(added);
    EXPECT_EQ(trajectory.size(), 1);
    EXPECT_TRUE(trajectory.empty());
}

TEST_F(trajectory_test, ProvidesAllPoses) {
    // GIVEN a trajectory with multiple motions
    visual_odometry::Trajectory trajectory;
    trajectory.add_motion(forward_motion_);
    trajectory.add_motion(forward_motion_);

    // WHEN getting all poses
    auto const& poses = trajectory.poses();

    // THEN should have 3 poses
    ASSERT_EQ(poses.size(), 3);

    // AND poses should be in sequence
    EXPECT_TRUE(poses[0].translation.isZero());
    EXPECT_NEAR(poses[1].translation(2), 1.0, 1e-9);
    EXPECT_NEAR(poses[2].translation(2), 2.0, 1e-9);
}

TEST_F(trajectory_test, ResetsToOrigin) {
    // GIVEN a trajectory with motions
    visual_odometry::Trajectory trajectory;
    trajectory.add_motion(forward_motion_);
    trajectory.add_motion(forward_motion_);

    // WHEN resetting
    trajectory.reset();

    // THEN trajectory should be at origin
    EXPECT_EQ(trajectory.size(), 1);
    EXPECT_TRUE(trajectory.empty());
    EXPECT_TRUE(trajectory.current_pose().translation.isZero());
}

TEST(PoseTest, IdentityPoseIsAtOrigin) {
    // GIVEN an identity pose
    auto const pose = visual_odometry::pose::identity();

    // THEN rotation should be identity and translation zero
    EXPECT_TRUE(pose.rotation.isIdentity());
    EXPECT_TRUE(pose.translation.isZero());
}

TEST(PoseTest, ComposeWithIdentityMotionIsUnchanged) {
    // GIVEN a pose
    visual_odometry::pose pose;
    pose.rotation = Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitY()).toRotationMatrix();
    pose.translation = Eigen::Vector3d(1, 2, 3);

    // AND an identity motion
    visual_odometry::motion_estimate identity_motion;
    identity_motion.rotation = Eigen::Matrix3d::Identity();
    identity_motion.translation = Eigen::Vector3d::Zero();
    identity_motion.valid = true;

    // WHEN composing
    auto const result = pose.compose(identity_motion);

    // THEN pose should be unchanged
    EXPECT_TRUE(result.rotation.isApprox(pose.rotation));
    EXPECT_TRUE(result.translation.isApprox(pose.translation));
}

TEST_F(trajectory_test, ToJsonProducesValidJson) {
    // GIVEN a trajectory with one motion
    visual_odometry::Trajectory trajectory;
    trajectory.add_motion(forward_motion_);

    // WHEN converting to JSON
    auto const json = trajectory.to_json();

    // THEN JSON should contain expected structure
    EXPECT_THAT(json, testing::HasSubstr("\"poses\""));
    EXPECT_THAT(json, testing::HasSubstr("\"rotation\""));
    EXPECT_THAT(json, testing::HasSubstr("\"translation\""));
}

TEST_F(trajectory_test, ToJsonContainsAllPoses) {
    // GIVEN a trajectory with multiple motions
    visual_odometry::Trajectory trajectory;
    trajectory.add_motion(forward_motion_);
    trajectory.add_motion(forward_motion_);

    // WHEN converting to JSON
    auto const json = trajectory.to_json();

    // THEN JSON should contain expected number of poses
    // Count occurrences of "rotation" to verify 3 poses
    size_t count = 0;
    std::string const search = "\"rotation\"";
    size_t pos = 0;
    while ((pos = json.find(search, pos)) != std::string::npos) {
        ++count;
        pos += search.length();
    }
    EXPECT_EQ(count, 3);
}

TEST_F(trajectory_test, SaveToJsonCreatesFile) {
    // GIVEN a trajectory
    visual_odometry::Trajectory trajectory;
    trajectory.add_motion(forward_motion_);

    // AND a temp file path
    auto const filepath = std::filesystem::temp_directory_path() / "test_trajectory.json";

    // WHEN saving to JSON
    auto const result = trajectory.save_to_json(filepath);

    // THEN save should succeed
    ASSERT_TRUE(result.has_value()) << result.error();

    // AND file should exist
    EXPECT_TRUE(std::filesystem::exists(filepath));

    // AND file should contain valid JSON
    std::ifstream file(filepath);
    std::string const content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    EXPECT_THAT(content, testing::HasSubstr("\"poses\""));

    // Cleanup
    std::filesystem::remove(filepath);
}

TEST_F(trajectory_test, SaveToJsonReturnsErrorForInvalidPath) {
    // GIVEN a trajectory
    visual_odometry::Trajectory const trajectory;

    // WHEN saving to an invalid path
    auto const result = trajectory.save_to_json("/nonexistent/directory/file.json");

    // THEN save should fail with error
    ASSERT_FALSE(result.has_value());
    EXPECT_THAT(result.error(), testing::HasSubstr("Failed to open"));
}
