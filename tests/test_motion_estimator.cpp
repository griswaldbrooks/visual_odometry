#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <visual_odometry/motion_estimator.hpp>
#include <random>
#include <cmath>

class MotionEstimatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // KITTI-like camera intrinsics
        intrinsics_.fx = 718.856;
        intrinsics_.fy = 718.856;
        intrinsics_.cx = 607.1928;
        intrinsics_.cy = 185.2157;
    }

    // Generate synthetic matched points with known motion
    void generate_synthetic_points(Eigen::Matrix3d const& R,
                                    Eigen::Vector3d const& t,
                                    int num_points,
                                    std::vector<cv::Point2f>& points1,
                                    std::vector<cv::Point2f>& points2) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist_x(100, 1100);
        std::uniform_real_distribution<double> dist_y(50, 300);
        std::uniform_real_distribution<double> dist_z(5, 50);

        Eigen::Matrix3d K;
        K << intrinsics_.fx, 0, intrinsics_.cx,
             0, intrinsics_.fy, intrinsics_.cy,
             0, 0, 1;

        for (int i = 0; i < num_points; ++i) {
            // Random 3D point
            Eigen::Vector3d P(dist_x(rng) - intrinsics_.cx,
                              dist_y(rng) - intrinsics_.cy,
                              dist_z(rng));
            P(0) *= P(2) / intrinsics_.fx;
            P(1) *= P(2) / intrinsics_.fy;

            // Project to first camera
            Eigen::Vector3d p1 = K * P;
            p1 /= p1(2);

            // Transform to second camera and project
            Eigen::Vector3d P2 = R * P + t;
            Eigen::Vector3d p2 = K * P2;
            p2 /= p2(2);

            points1.emplace_back(static_cast<float>(p1(0)), static_cast<float>(p1(1)));
            points2.emplace_back(static_cast<float>(p2(0)), static_cast<float>(p2(1)));
        }
    }

    visual_odometry::CameraIntrinsics intrinsics_;
};

TEST_F(MotionEstimatorTest, EstimatesForwardMotion) {
    // Pure forward translation
    Eigen::Matrix3d const R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t(0, 0, 1);
    t.normalize();

    std::vector<cv::Point2f> points1, points2;
    generate_synthetic_points(R, t, 100, points1, points2);

    visual_odometry::MotionEstimator estimator(intrinsics_);
    auto const result = estimator.estimate(points1, points2);

    EXPECT_TRUE(result.valid);
    EXPECT_GT(result.inliers, 50);

    // Check translation direction (should be roughly forward)
    EXPECT_GT(result.translation(2), 0.9);
}

TEST_F(MotionEstimatorTest, EstimatesRotation) {
    // Small rotation around Y axis
    double const angle = 0.05;  // ~3 degrees
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY());
    Eigen::Vector3d t(0.1, 0, 1);
    t.normalize();

    std::vector<cv::Point2f> points1, points2;
    generate_synthetic_points(R, t, 100, points1, points2);

    visual_odometry::MotionEstimator estimator(intrinsics_);
    auto const result = estimator.estimate(points1, points2);

    EXPECT_TRUE(result.valid);
    EXPECT_GT(result.inliers, 50);
}

TEST_F(MotionEstimatorTest, FailsWithTooFewPoints) {
    std::vector<cv::Point2f> const points1 = {{100, 100}, {200, 200}, {300, 300}};
    std::vector<cv::Point2f> const points2 = {{110, 100}, {210, 200}, {310, 300}};

    visual_odometry::MotionEstimator estimator(intrinsics_);
    auto const result = estimator.estimate(points1, points2);

    EXPECT_FALSE(result.valid);
}

TEST_F(MotionEstimatorTest, CameraMatrixIsCorrect) {
    cv::Mat const K = intrinsics_.camera_matrix();

    EXPECT_DOUBLE_EQ(K.at<double>(0, 0), intrinsics_.fx);
    EXPECT_DOUBLE_EQ(K.at<double>(1, 1), intrinsics_.fy);
    EXPECT_DOUBLE_EQ(K.at<double>(0, 2), intrinsics_.cx);
    EXPECT_DOUBLE_EQ(K.at<double>(1, 2), intrinsics_.cy);
    EXPECT_DOUBLE_EQ(K.at<double>(2, 2), 1.0);
}

TEST_F(MotionEstimatorTest, LoadsIntrinsicsFromYaml) {
    // This test requires the KITTI camera file to exist
    std::string const camera_file = "data/kitti_camera.yaml";

    auto const loaded = visual_odometry::CameraIntrinsics::load_from_yaml(camera_file);
    if (!loaded.has_value()) {
        GTEST_SKIP() << "Camera file not found, skipping YAML load test: " << loaded.error();
    }

    EXPECT_NEAR(loaded.value().fx, 718.856, 0.001);
    EXPECT_NEAR(loaded.value().fy, 718.856, 0.001);
    EXPECT_NEAR(loaded.value().cx, 607.1928, 0.001);
    EXPECT_NEAR(loaded.value().cy, 185.2157, 0.001);
}

TEST_F(MotionEstimatorTest, LoadFromYamlReturnsErrorForMissingFile) {
    auto const result = visual_odometry::CameraIntrinsics::load_from_yaml("/nonexistent/path.yaml");

    ASSERT_FALSE(result.has_value());
    EXPECT_THAT(result.error(), testing::HasSubstr("Could not open"));
}
