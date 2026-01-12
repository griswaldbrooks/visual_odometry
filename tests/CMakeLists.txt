add_executable(
  visual_odometry_tests
  test_main.cpp
)

target_link_libraries(
  visual_odometry_tests
  PRIVATE
    visual_odometry::lib
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
)

include(CTest)
include(GoogleTest)
gtest_discover_tests(visual_odometry_tests)
