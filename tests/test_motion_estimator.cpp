#include <gtest/gtest.h>
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
    void generateSyntheticPoints(const Eigen::Matrix3d& R,
                                  const Eigen::Vector3d& t,
                                  int numPoints,
                                  std::vector<cv::Point2f>& points1,
                                  std::vector<cv::Point2f>& points2) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> distX(100, 1100);
        std::uniform_real_distribution<double> distY(50, 300);
        std::uniform_real_distribution<double> distZ(5, 50);

        Eigen::Matrix3d K;
        K << intrinsics_.fx, 0, intrinsics_.cx,
             0, intrinsics_.fy, intrinsics_.cy,
             0, 0, 1;

        for (int i = 0; i < numPoints; ++i) {
            // Random 3D point
            Eigen::Vector3d P(distX(rng) - intrinsics_.cx,
                              distY(rng) - intrinsics_.cy,
                              distZ(rng));
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
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t(0, 0, 1);
    t.normalize();

    std::vector<cv::Point2f> points1, points2;
    generateSyntheticPoints(R, t, 100, points1, points2);

    visual_odometry::MotionEstimator estimator(intrinsics_);
    auto result = estimator.estimate(points1, points2);

    EXPECT_TRUE(result.valid);
    EXPECT_GT(result.inliers, 50);

    // Check translation direction (should be roughly forward)
    EXPECT_GT(result.translation(2), 0.9);
}

TEST_F(MotionEstimatorTest, EstimatesRotation) {
    // Small rotation around Y axis
    double angle = 0.05;  // ~3 degrees
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY());
    Eigen::Vector3d t(0.1, 0, 1);
    t.normalize();

    std::vector<cv::Point2f> points1, points2;
    generateSyntheticPoints(R, t, 100, points1, points2);

    visual_odometry::MotionEstimator estimator(intrinsics_);
    auto result = estimator.estimate(points1, points2);

    EXPECT_TRUE(result.valid);
    EXPECT_GT(result.inliers, 50);
}

TEST_F(MotionEstimatorTest, FailsWithTooFewPoints) {
    std::vector<cv::Point2f> points1 = {{100, 100}, {200, 200}, {300, 300}};
    std::vector<cv::Point2f> points2 = {{110, 100}, {210, 200}, {310, 300}};

    visual_odometry::MotionEstimator estimator(intrinsics_);
    auto result = estimator.estimate(points1, points2);

    EXPECT_FALSE(result.valid);
}

TEST_F(MotionEstimatorTest, CameraMatrixIsCorrect) {
    cv::Mat K = intrinsics_.cameraMatrix();

    EXPECT_DOUBLE_EQ(K.at<double>(0, 0), intrinsics_.fx);
    EXPECT_DOUBLE_EQ(K.at<double>(1, 1), intrinsics_.fy);
    EXPECT_DOUBLE_EQ(K.at<double>(0, 2), intrinsics_.cx);
    EXPECT_DOUBLE_EQ(K.at<double>(1, 2), intrinsics_.cy);
    EXPECT_DOUBLE_EQ(K.at<double>(2, 2), 1.0);
}

TEST_F(MotionEstimatorTest, LoadsIntrinsicsFromYaml) {
    // This test requires the KITTI camera file to exist
    std::string cameraFile = "data/kitti_camera.yaml";

    try {
        auto loaded = visual_odometry::CameraIntrinsics::loadFromYaml(cameraFile);
        EXPECT_NEAR(loaded.fx, 718.856, 0.001);
        EXPECT_NEAR(loaded.fy, 718.856, 0.001);
        EXPECT_NEAR(loaded.cx, 607.1928, 0.001);
        EXPECT_NEAR(loaded.cy, 185.2157, 0.001);
    } catch (const std::runtime_error&) {
        GTEST_SKIP() << "Camera file not found, skipping YAML load test";
    }
}
