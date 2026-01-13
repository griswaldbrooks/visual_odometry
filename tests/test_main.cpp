#include <gtest/gtest.h>
#include <visual_odometry/visual_odometry.hpp>

TEST(VisualOdometryTest, CanBeConstructed) {
    // GIVEN nothing (default construction)
    // WHEN constructing a VisualOdometry object
    visual_odometry::VisualOdometry vo;

    // THEN it should construct successfully
    EXPECT_TRUE(true);
}
