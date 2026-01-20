cmake_minimum_required(VERSION 3.25)

project(
  visual_odometry
  VERSION 0.1.0
  DESCRIPTION "Visual odometry system using feature-based motion estimation"
  LANGUAGES CXX
)

# Only do these if this is the main project
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # Support folders in IDEs
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  # Export compile commands for clang-tidy
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  # Set RPATH to find libraries in conda/pixi environment
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
  if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
  endif()
endif()

# Options
option(ENABLE_TESTING "Enable testing" ON)
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(ENABLE_SANITIZERS "Enable sanitizers (ASan, UBSan)" OFF)
option(ENABLE_CLANG_TIDY "Enable clang-tidy checks" OFF)
option(ENABLE_DEVELOPER_MODE "Enable developer mode" ON)
option(ENABLE_PYTHON_BINDINGS "Enable Python bindings via nanobind" OFF)

# Find packages
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Fetch tl::expected (header-only)
include(FetchContent)
FetchContent_Declare(
  tl-expected
  GIT_REPOSITORY https://github.com/TartanLlama/expected.git
  GIT_TAG v1.1.0
)
FetchContent_MakeAvailable(tl-expected)

# Python bindings dependencies
if(ENABLE_PYTHON_BINDINGS)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

  # Fetch nanobind
  FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.4.0
  )
  FetchContent_MakeAvailable(nanobind)

  # Fetch cvnp_nano for cv::Mat <-> numpy conversion (header-only, don't run CMakeLists)
  FetchContent_Declare(
    cvnp_nano
    GIT_REPOSITORY https://github.com/pthom/cvnp_nano.git
    GIT_TAG main
  )
  FetchContent_Populate(cvnp_nano)
  # cvnp_nano is header-only, we just need the include path: ${cvnp_nano_SOURCE_DIR}
endif()

# Compiler warnings
add_library(project_warnings INTERFACE)
target_compile_options(
  project_warnings
  INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wconversion
    -Wsign-conversion
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
)

# Sanitizers
add_library(project_sanitizers INTERFACE)
if(ENABLE_SANITIZERS)
  message(STATUS "Enabling sanitizers: AddressSanitizer, UndefinedBehaviorSanitizer")
  target_compile_options(
    project_sanitizers
    INTERFACE
      -fsanitize=address,undefined
      -fno-omit-frame-pointer
      -fno-optimize-sibling-calls
  )
  target_link_options(
    project_sanitizers
    INTERFACE
      -fsanitize=address,undefined
  )
endif()

# Coverage
add_library(project_coverage INTERFACE)
if(ENABLE_COVERAGE)
  message(STATUS "Enabling coverage reporting")
  target_compile_options(
    project_coverage
    INTERFACE
      --coverage
      -fprofile-arcs
      -ftest-coverage
  )
  target_link_options(
    project_coverage
    INTERFACE
      --coverage
  )
endif()

# Clang-tidy
if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES "clang-tidy" REQUIRED)
  message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
  set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif()

# Project library
add_library(visual_odometry_lib
  src/visual_odometry.cpp
  src/image_loader.cpp
  src/feature_detector.cpp
  src/feature_matcher.cpp
  src/image_matcher.cpp
  src/motion_estimator.cpp
  src/trajectory.cpp
)
add_library(visual_odometry::lib ALIAS visual_odometry_lib)

target_sources(
  visual_odometry_lib
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS include
    FILES
      include/visual_odometry/visual_odometry.hpp
      include/visual_odometry/image_loader.hpp
      include/visual_odometry/feature_detector.hpp
      include/visual_odometry/feature_matcher.hpp
      include/visual_odometry/image_matcher.hpp
      include/visual_odometry/motion_estimator.hpp
      include/visual_odometry/trajectory.hpp
)

target_compile_features(visual_odometry_lib PUBLIC cxx_std_20)

set_target_properties(
  visual_odometry_lib
  PROPERTIES
    CXX_EXTENSIONS OFF
)

target_link_libraries(
  visual_odometry_lib
  PUBLIC
    project_warnings
    project_sanitizers
    project_coverage
    ${OpenCV_LIBS}
    Eigen3::Eigen
    tl::expected
)

target_include_directories(
  visual_odometry_lib
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)

# Main executable (not built during coverage builds to avoid gcov issues)
if(NOT ENABLE_COVERAGE)
  add_executable(visual_odometry src/main.cpp)
  target_link_libraries(visual_odometry PRIVATE visual_odometry::lib)
endif()

# Testing
if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

# Doxygen
if(ENABLE_DEVELOPER_MODE)
  find_package(Doxygen)
  if(Doxygen_FOUND)
    set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docs")
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_GENERATE_MAN NO)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
    set(DOXYGEN_RECURSIVE YES)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

    doxygen_add_docs(
      docs
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/README.md
      COMMENT "Generating API documentation with Doxygen"
    )
  else()
    message(STATUS "Doxygen not found, not building docs")
  endif()
endif()

# Coverage targets
if(ENABLE_COVERAGE)
  add_custom_target(
    coverage
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
    COMMAND gcovr
      --gcov-executable "llvm-cov gcov"
      --root ${CMAKE_SOURCE_DIR}
      --filter ${CMAKE_SOURCE_DIR}/src
      --filter ${CMAKE_SOURCE_DIR}/include
      --exclude ${CMAKE_SOURCE_DIR}/tests
      --xml ${CMAKE_BINARY_DIR}/coverage/coverage.xml
      --html-details ${CMAKE_BINARY_DIR}/coverage/coverage.html
      --print-summary
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating coverage report"
    DEPENDS visual_odometry_tests
  )
endif()

# Python bindings
if(ENABLE_PYTHON_BINDINGS)
  # Create a version of the library without sanitizers for Python bindings
  add_library(visual_odometry_lib_py
    src/visual_odometry.cpp
    src/image_loader.cpp
    src/feature_detector.cpp
    src/feature_matcher.cpp
    src/image_matcher.cpp
    src/motion_estimator.cpp
    src/trajectory.cpp
  )

  target_compile_features(visual_odometry_lib_py PUBLIC cxx_std_20)
  set_target_properties(visual_odometry_lib_py PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)

  target_link_libraries(
    visual_odometry_lib_py
    PUBLIC
      ${OpenCV_LIBS}
      Eigen3::Eigen
      tl::expected
  )

  target_include_directories(
    visual_odometry_lib_py
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${OpenCV_INCLUDE_DIRS}
  )

  # Create Python module
  nanobind_add_module(
    _visual_odometry_impl
    NB_STATIC
    src/python_bindings.cpp
  )

  target_link_libraries(
    _visual_odometry_impl
    PRIVATE
      visual_odometry_lib_py
  )

  target_include_directories(
    _visual_odometry_impl
    PRIVATE
      ${cvnp_nano_SOURCE_DIR}
  )

  # Install the module to the python package directory
  install(TARGETS _visual_odometry_impl LIBRARY DESTINATION visual_odometry)
endif()
