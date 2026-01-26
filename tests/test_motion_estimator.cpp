#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <visual_odometry/motion_estimator.hpp>

class motion_estimator_test : public ::testing::Test {
protected:
    void SetUp() override {
        // KITTI-like camera intrinsics
        intrinsics_.fx = 718.856;
        intrinsics_.fy = 718.856;
        intrinsics_.cx = 607.1928;
        intrinsics_.cy = 185.2157;
    }

    // Generate synthetic matched points with known motion
    void generate_synthetic_points(
        Eigen::Matrix3d const& rotation,     // NOLINT(misc-include-cleaner)
        Eigen::Vector3d const& translation,  // NOLINT(misc-include-cleaner)
        int num_points, std::vector<cv::Point2f>& points1,
        std::vector<cv::Point2f>& points2) const {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist_x(100, 1100);
        std::uniform_real_distribution<double> dist_y(50, 300);
        std::uniform_real_distribution<double> dist_z(5, 50);

        Eigen::Matrix3d intrinsics_matrix;
        intrinsics_matrix << intrinsics_.fx, 0, intrinsics_.cx, 0, intrinsics_.fy, intrinsics_.cy,
            0, 0, 1;

        for (int i = 0; i < num_points; ++i) {
            // Random 3D point
            Eigen::Vector3d point_3d(dist_x(rng) - intrinsics_.cx, dist_y(rng) - intrinsics_.cy,
                                     dist_z(rng));
            point_3d(0) *= point_3d(2) / intrinsics_.fx;
            point_3d(1) *= point_3d(2) / intrinsics_.fy;

            // Project to first camera
            Eigen::Vector3d p1 = intrinsics_matrix * point_3d;
            p1 /= p1(2);

            // Transform to second camera and project
            Eigen::Vector3d const transformed_point = rotation * point_3d + translation;
            Eigen::Vector3d p2 = intrinsics_matrix * transformed_point;
            p2 /= p2(2);

            points1.emplace_back(static_cast<float>(p1(0)), static_cast<float>(p1(1)));
            points2.emplace_back(static_cast<float>(p2(0)), static_cast<float>(p2(1)));
        }
    }

    visual_odometry::camera_intrinsics intrinsics_{.fx = 0.0, .fy = 0.0, .cx = 0.0, .cy = 0.0};
};

TEST_F(motion_estimator_test, EstimatesForwardMotion) {
    // GIVEN synthetic points with pure forward translation
    Eigen::Matrix3d const rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation(0, 0, 1);
    translation.normalize();

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    generate_synthetic_points(rotation, translation, 100, points1, points2);

    // WHEN estimating motion
    visual_odometry::motion_estimator_config const config{};
    auto const result = visual_odometry::estimate_motion(points1, points2, intrinsics_, config);

    // THEN estimation should succeed
    EXPECT_TRUE(result.valid);
    EXPECT_GT(result.inliers, 50);

    // AND translation should be roughly forward
    EXPECT_GT(result.translation(2), 0.9);
}

TEST_F(motion_estimator_test, EstimatesRotation) {
    // GIVEN synthetic points with small rotation around Y axis
    double const angle = 0.05;  // ~3 degrees
    Eigen::Matrix3d rotation;
    rotation = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY());
    Eigen::Vector3d translation(0.1, 0, 1);
    translation.normalize();

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    generate_synthetic_points(rotation, translation, 100, points1, points2);

    // WHEN estimating motion
    visual_odometry::motion_estimator_config const config{};
    auto const result = visual_odometry::estimate_motion(points1, points2, intrinsics_, config);

    // THEN estimation should succeed
    EXPECT_TRUE(result.valid);
    EXPECT_GT(result.inliers, 50);
}

TEST_F(motion_estimator_test, FailsWithTooFewPoints) {
    // GIVEN fewer than 5 points
    std::vector<cv::Point2f> const points1 = {{100, 100}, {200, 200}, {300, 300}};
    std::vector<cv::Point2f> const points2 = {{110, 100}, {210, 200}, {310, 300}};

    // WHEN estimating motion
    visual_odometry::motion_estimator_config const config{};
    auto const result = visual_odometry::estimate_motion(points1, points2, intrinsics_, config);

    // THEN estimation should fail
    EXPECT_FALSE(result.valid);
}

TEST_F(motion_estimator_test, CameraMatrixIsCorrect) {
    // GIVEN camera intrinsics
    // WHEN getting the camera matrix
    cv::Mat const intrinsics_matrix = intrinsics_.camera_matrix();

    // THEN matrix should contain correct values
    EXPECT_DOUBLE_EQ(intrinsics_matrix.at<double>(0, 0), intrinsics_.fx);
    EXPECT_DOUBLE_EQ(intrinsics_matrix.at<double>(1, 1), intrinsics_.fy);
    EXPECT_DOUBLE_EQ(intrinsics_matrix.at<double>(0, 2), intrinsics_.cx);
    EXPECT_DOUBLE_EQ(intrinsics_matrix.at<double>(1, 2), intrinsics_.cy);
    EXPECT_DOUBLE_EQ(intrinsics_matrix.at<double>(2, 2), 1.0);
}

TEST_F(motion_estimator_test, LoadsIntrinsicsFromYaml) {
    // GIVEN a path to a camera YAML file
    std::string const camera_file = "data/kitti_camera.yaml";

    // WHEN loading intrinsics
    auto const loaded = visual_odometry::camera_intrinsics::load_from_yaml(camera_file);
    if (!loaded.has_value()) {
        GTEST_SKIP() << "Camera file not found, skipping YAML load test: " << loaded.error();
    }

    // THEN intrinsics should match expected KITTI values
    EXPECT_NEAR(loaded.value().fx, 718.856, 0.001);
    EXPECT_NEAR(loaded.value().fy, 718.856, 0.001);
    EXPECT_NEAR(loaded.value().cx, 607.1928, 0.001);
    EXPECT_NEAR(loaded.value().cy, 185.2157, 0.001);
}

TEST_F(motion_estimator_test, LoadFromYamlReturnsErrorForMissingFile) {
    // GIVEN a nonexistent file path
    // WHEN loading intrinsics
    auto const result =
        visual_odometry::camera_intrinsics::load_from_yaml("/nonexistent/path.yaml");

    // THEN loading should fail with error
    ASSERT_FALSE(result.has_value());
    EXPECT_THAT(result.error(), testing::HasSubstr("Could not open"));
}
