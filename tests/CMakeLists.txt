add_executable(
  visual_odometry_tests
  test_main.cpp
)

target_link_libraries(
  visual_odometry_tests
  PRIVATE
    visual_odometry::lib
    Catch2::Catch2WithMain
)

include(CTest)
include(Catch)
catch_discover_tests(visual_odometry_tests)
