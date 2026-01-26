/**
 * @file python_bindings.cpp
 * @brief Python bindings for visual_odometry library using nanobind.
 */

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

// cvnp_nano for cv::Mat <-> numpy conversion
#include <cvnp_nano/cvnp_nano.h>
#include <visual_odometry/image_loader.hpp>
#include <visual_odometry/image_matcher.hpp>
#include <visual_odometry/motion_estimator.hpp>
#include <visual_odometry/trajectory.hpp>

namespace nb = nanobind;
using namespace nb::literals;
namespace vo = visual_odometry;

// Helper to convert tl::expected errors to Python exceptions
template <typename T>
T unwrap_expected(tl::expected<T, std::string>&& result) {
    if (!result.has_value()) {
        throw std::runtime_error(result.error());
    }
    return std::move(result.value());
}

// Specialization for void
inline void unwrap_expected_void(tl::expected<void, std::string>&& result) {
    if (!result.has_value()) {
        throw std::runtime_error(result.error());
    }
}

NB_MODULE(_visual_odometry_impl, m) {
    m.doc() = "Visual odometry library - Python bindings";
    m.attr("__version__") = "0.1.0";

    // ==================== image_loader ====================
    nb::class_<vo::image_loader>(m, "image_loader",
                                 "Loads sequential images from a directory for visual odometry.")

        .def_static(
            "create",
            [](std::string_view path) { return unwrap_expected(vo::image_loader::create(path)); },
            "path"_a,
            "Create a new image_loader from a directory path.\n\n"
            "Args:\n"
            "    path: Path to directory containing images.\n\n"
            "Returns:\n"
            "    image_loader instance.\n\n"
            "Raises:\n"
            "    RuntimeError: If directory does not exist.")

        .def(
            "load_image",
            [](vo::image_loader const& self, size_t index) {
                return unwrap_expected(self.load_image(index));
            },
            "index"_a,
            "Load a single image by index.\n\n"
            "Args:\n"
            "    index: Image index (0-based).\n\n"
            "Returns:\n"
            "    Grayscale image as numpy array.")

        .def(
            "load_image_pair",
            [](vo::image_loader const& self, size_t index) {
                return unwrap_expected(self.load_image_pair(index));
            },
            "index"_a,
            "Load a pair of consecutive images.\n\n"
            "Args:\n"
            "    index: Index of first image.\n\n"
            "Returns:\n"
            "    Tuple of (image[index], image[index+1]).")

        .def(
            "next_pair", [](vo::image_loader& self) { return unwrap_expected(self.next_pair()); },
            "Get the next pair of images and advance the index.\n\n"
            "Returns:\n"
            "    Tuple of consecutive images.")

        .def("has_next", &vo::image_loader::has_next, "Check if more image pairs are available.")

        .def("reset", &vo::image_loader::reset, "Reset to the first image.")

        .def("size", &vo::image_loader::size, "Get total number of images.")

        .def("__len__", &vo::image_loader::size)

        .def(
            "__iter__",
            [](vo::image_loader& self) -> vo::image_loader& {
                self.reset();
                return self;
            },
            nb::rv_policy::reference)

        .def("__next__", [](vo::image_loader& self) {
            if (!self.has_next()) {
                throw nb::stop_iteration();
            }
            return unwrap_expected(self.next_pair());
        });

    // ==================== MatchResult ====================
    nb::class_<vo::match_result>(m, "MatchResult", "Result of feature matching between two images.")

        .def_ro("points1", &vo::match_result::points1,
                "Matched points in first image as list of (x, y) tuples.")

        .def_ro("points2", &vo::match_result::points2,
                "Matched points in second image as list of (x, y) tuples.")

        .def_prop_ro(
            "num_matches", [](vo::match_result const& self) { return self.matches.size(); },
            "Number of matches found.")

        .def("__len__", [](vo::match_result const& self) { return self.matches.size(); });

    // ==================== orb_matcher ====================
    nb::class_<vo::orb_matcher>(m, "orb_matcher", "ORB-based image matcher (detect + match).")

        .def(nb::init<int, float>(), "max_features"_a = vo::default_max_features,
             "ratio_threshold"_a = vo::default_ratio_threshold,
             "Construct ORB matcher.\n\n"
             "Args:\n"
             "    max_features: Maximum features to detect (default: 2000).\n"
             "    ratio_threshold: Lowe's ratio test threshold (default: 0.75).")

        .def("match_images", &vo::orb_matcher::match_images, "img1"_a, "img2"_a,
             "Match features between two images.\n\n"
             "Args:\n"
             "    img1: First grayscale image (numpy array).\n"
             "    img2: Second grayscale image (numpy array).\n\n"
             "Returns:\n"
             "    MatchResult containing corresponding points.")

        .def_prop_ro("name", &vo::orb_matcher::name, "Name of this matcher backend.");

    // ==================== lightglue_matcher ====================
    nb::class_<vo::lightglue_matcher>(m, "lightglue_matcher",
                                      "LightGlue learned feature matcher using ONNX Runtime.")

        .def(nb::init<std::filesystem::path>(),
             "model_path"_a = "models/disk_lightglue_end2end.onnx",
             "Construct LightGlue matcher.\n\n"
             "Args:\n"
             "    model_path: Path to ONNX model file (default: "
             "models/disk_lightglue_end2end.onnx).")

        .def("match_images", &vo::lightglue_matcher::match_images, "img1"_a, "img2"_a,
             "Match features between two images.\n\n"
             "Args:\n"
             "    img1: First grayscale image (numpy array).\n"
             "    img2: Second grayscale image (numpy array).\n\n"
             "Returns:\n"
             "    MatchResult containing corresponding points.")

        .def_prop_ro("name", &vo::lightglue_matcher::name, "Name of this matcher backend.");

    // ==================== CameraIntrinsics ====================
    nb::class_<vo::camera_intrinsics>(m, "CameraIntrinsics", "Camera intrinsic parameters.")

        .def(nb::init<>())

        .def_rw("fx", &vo::camera_intrinsics::fx, "Focal length x.")
        .def_rw("fy", &vo::camera_intrinsics::fy, "Focal length y.")
        .def_rw("cx", &vo::camera_intrinsics::cx, "Principal point x.")
        .def_rw("cy", &vo::camera_intrinsics::cy, "Principal point y.")

        .def_static(
            "load_from_yaml",
            [](std::string_view filepath) {
                return unwrap_expected(vo::camera_intrinsics::load_from_yaml(filepath));
            },
            "filepath"_a,
            "Load intrinsics from YAML file.\n\n"
            "Args:\n"
            "    filepath: Path to YAML file.\n\n"
            "Returns:\n"
            "    CameraIntrinsics instance.")

        .def("__repr__", [](vo::camera_intrinsics const& self) {
            return "CameraIntrinsics(fx=" + std::to_string(self.fx) +
                   ", fy=" + std::to_string(self.fy) + ", cx=" + std::to_string(self.cx) +
                   ", cy=" + std::to_string(self.cy) + ")";
        });

    // ==================== MotionEstimate ====================
    nb::class_<vo::motion_estimate>(m, "MotionEstimate",
                                    "Result of motion estimation between two frames.")

        .def_ro("rotation", &vo::motion_estimate::rotation, "Rotation matrix (3x3 numpy array).")

        .def_ro("translation", &vo::motion_estimate::translation,
                "Translation vector (3-element numpy array), unit norm.")

        .def_ro("inliers", &vo::motion_estimate::inliers, "Number of RANSAC inliers.")

        .def_ro("valid", &vo::motion_estimate::valid, "Whether estimation succeeded.")

        .def("__bool__", [](vo::motion_estimate const& self) { return self.valid; });

    // ==================== MotionEstimatorConfig ====================
    nb::class_<vo::motion_estimator_config>(m, "MotionEstimatorConfig",
                                            "Configuration parameters for motion estimation.")

        .def(nb::init<>())

        .def_rw("ransac_threshold", &vo::motion_estimator_config::ransac_threshold,
                "RANSAC reprojection threshold in pixels (default: 1.0).")

        .def_rw("ransac_confidence", &vo::motion_estimator_config::ransac_confidence,
                "RANSAC confidence level 0-1 (default: 0.999).")

        .def("__repr__", [](vo::motion_estimator_config const& self) {
            return "MotionEstimatorConfig(ransac_threshold=" +
                   std::to_string(self.ransac_threshold) +
                   ", ransac_confidence=" + std::to_string(self.ransac_confidence) + ")";
        });

    // ==================== estimate_motion function ====================
    m.def(
        "estimate_motion",
        [](std::vector<cv::Point2f> const& points1, std::vector<cv::Point2f> const& points2,
           vo::camera_intrinsics const& intrinsics, vo::motion_estimator_config const& config) {
            return vo::estimate_motion(points1, points2, intrinsics, config);
        },
        "points1"_a, "points2"_a, "intrinsics"_a, "config"_a = vo::motion_estimator_config{},
        "Estimate camera motion from matched feature points.\n\n"
        "Args:\n"
        "    points1: Points in first image.\n"
        "    points2: Corresponding points in second image.\n"
        "    intrinsics: Camera intrinsic parameters.\n"
        "    config: Configuration for RANSAC parameters (optional).\n\n"
        "Returns:\n"
        "    MotionEstimate containing R, t if successful.");

    // ==================== Pose ====================
    nb::class_<vo::pose>(m, "Pose", "Represents an absolute camera pose in the world frame.")

        .def(nb::init<>())

        .def_rw("rotation", &vo::pose::rotation, "Rotation matrix (3x3 numpy array).")

        .def_rw("translation", &vo::pose::translation,
                "Translation vector (3-element numpy array).")

        .def_static("identity", &vo::pose::identity, "Create identity pose (origin).")

        .def("compose", &vo::pose::compose, "motion"_a,
             "Compose this pose with a relative transform.\n\n"
             "Args:\n"
             "    motion: Relative motion (MotionEstimate) from this pose.\n\n"
             "Returns:\n"
             "    New absolute pose after applying the transform.");

    // ==================== Trajectory ====================
    nb::class_<vo::Trajectory>(m, "Trajectory", "Accumulates camera poses to build a trajectory.")

        .def(nb::init<>(), "Construct empty trajectory starting at origin.")

        .def("add_motion", &vo::Trajectory::add_motion, "motion"_a,
             "Add a relative motion estimate to the trajectory.\n\n"
             "Args:\n"
             "    motion: Relative motion from current to next frame.\n\n"
             "Returns:\n"
             "    True if motion was valid and added, False otherwise.")

        .def_prop_ro("poses", &vo::Trajectory::poses, "Get all poses in the trajectory.")

        .def_prop_ro("current_pose", &vo::Trajectory::current_pose,
                     "Get the current (latest) pose.")

        .def("size", &vo::Trajectory::size, "Get the number of poses in the trajectory.")

        .def("empty", &vo::Trajectory::empty, "Check if trajectory is empty (only has origin).")

        .def("reset", &vo::Trajectory::reset, "Reset trajectory to origin.")

        .def(
            "save_to_json",
            [](vo::Trajectory const& self, std::string_view filepath) {
                unwrap_expected_void(self.save_to_json(std::filesystem::path(filepath)));
            },
            "filepath"_a, "Save trajectory to JSON file.")

        .def("to_json", &vo::Trajectory::to_json, "Convert trajectory to JSON string.")

        .def("__len__", &vo::Trajectory::size)

        .def(
            "__getitem__",
            [](vo::Trajectory const& self, size_t index) {
                if (index >= self.size()) {
                    throw nb::index_error();
                }
                return self.poses()[index];
            },
            "index"_a);
}
