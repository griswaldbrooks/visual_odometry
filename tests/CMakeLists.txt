add_executable(
  visual_odometry_tests
  test_main.cpp
  test_image_loader.cpp
  test_feature_detector.cpp
  test_feature_matcher.cpp
  test_motion_estimator.cpp
  test_trajectory.cpp
  $<$<BOOL:${ONNXRUNTIME_FOUND}>:test_onnx_session.cpp>
  $<$<BOOL:${ONNXRUNTIME_FOUND}>:test_lightglue_matcher.cpp>)

target_link_libraries(
  visual_odometry_tests PRIVATE visual_odometry::lib GTest::gtest
                                GTest::gtest_main GTest::gmock)

include(CTest)
include(GoogleTest)

gtest_discover_tests(
  visual_odometry_tests
  PROPERTIES
    ENVIRONMENT
    "LSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/lsan_suppressions.txt"
)
