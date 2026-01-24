#include <gtest/gtest.h>
#include <visual_odometry/visual_odometry.hpp>

TEST(visual_odometry_pipelineTest, CanBeConstructed) {
    // GIVEN nothing (default construction)
    // WHEN constructing a visual_odometry_pipeline object
    visual_odometry::visual_odometry_pipeline const vo;

    // THEN it should construct successfully
    EXPECT_TRUE(true);
}
