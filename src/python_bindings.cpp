/**
 * @file python_bindings.cpp
 * @brief Python bindings for visual_odometry library using nanobind.
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>

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
template<typename T>
T unwrap_expected(tl::expected<T, std::string>&& result) {
    if (!result) {
        throw std::runtime_error(result.error());
    }
    return std::move(*result);
}

// Specialization for void
inline void unwrap_expected_void(tl::expected<void, std::string>&& result) {
    if (!result) {
        throw std::runtime_error(result.error());
    }
}

NB_MODULE(_visual_odometry_impl, m) {
    m.doc() = "Visual odometry library - Python bindings";
    m.attr("__version__") = "0.1.0";

    // ==================== ImageLoader ====================
    nb::class_<vo::ImageLoader>(m, "ImageLoader",
        "Loads sequential images from a directory for visual odometry.")

        .def_static("create", [](std::string_view path) {
            return unwrap_expected(vo::ImageLoader::create(path));
        }, "path"_a,
        "Create a new ImageLoader from a directory path.\n\n"
        "Args:\n"
        "    path: Path to directory containing images.\n\n"
        "Returns:\n"
        "    ImageLoader instance.\n\n"
        "Raises:\n"
        "    RuntimeError: If directory does not exist.")

        .def("load_image", [](vo::ImageLoader const& self, size_t index) {
            return unwrap_expected(self.load_image(index));
        }, "index"_a,
        "Load a single image by index.\n\n"
        "Args:\n"
        "    index: Image index (0-based).\n\n"
        "Returns:\n"
        "    Grayscale image as numpy array.")

        .def("load_image_pair", [](vo::ImageLoader const& self, size_t index) {
            return unwrap_expected(self.load_image_pair(index));
        }, "index"_a,
        "Load a pair of consecutive images.\n\n"
        "Args:\n"
        "    index: Index of first image.\n\n"
        "Returns:\n"
        "    Tuple of (image[index], image[index+1]).")

        .def("next_pair", [](vo::ImageLoader& self) {
            return unwrap_expected(self.next_pair());
        },
        "Get the next pair of images and advance the index.\n\n"
        "Returns:\n"
        "    Tuple of consecutive images.")

        .def("has_next", &vo::ImageLoader::has_next,
        "Check if more image pairs are available.")

        .def("reset", &vo::ImageLoader::reset,
        "Reset to the first image.")

        .def("size", &vo::ImageLoader::size,
        "Get total number of images.")

        .def("__len__", &vo::ImageLoader::size)

        .def("__iter__", [](vo::ImageLoader& self) -> vo::ImageLoader& {
            self.reset();
            return self;
        }, nb::rv_policy::reference)

        .def("__next__", [](vo::ImageLoader& self) {
            if (!self.has_next()) {
                throw nb::stop_iteration();
            }
            return unwrap_expected(self.next_pair());
        });

    // ==================== MatchResult ====================
    nb::class_<vo::MatchResult>(m, "MatchResult",
        "Result of feature matching between two images.")

        .def_ro("points1", &vo::MatchResult::points1,
        "Matched points in first image as list of (x, y) tuples.")

        .def_ro("points2", &vo::MatchResult::points2,
        "Matched points in second image as list of (x, y) tuples.")

        .def_prop_ro("num_matches", [](vo::MatchResult const& self) {
            return self.matches.size();
        }, "Number of matches found.")

        .def("__len__", [](vo::MatchResult const& self) {
            return self.matches.size();
        });

    // ==================== ImageMatcher ====================
    nb::class_<vo::ImageMatcher>(m, "ImageMatcher",
        "Abstract interface for image-to-image feature matching.")

        .def("match_images", &vo::ImageMatcher::match_images,
        "img1"_a, "img2"_a,
        "Match features between two images.\n\n"
        "Args:\n"
        "    img1: First grayscale image (numpy array).\n"
        "    img2: Second grayscale image (numpy array).\n\n"
        "Returns:\n"
        "    MatchResult containing corresponding points.")

        .def_prop_ro("name", &vo::ImageMatcher::name,
        "Name of this matcher backend.");

    // ==================== OrbImageMatcher ====================
    nb::class_<vo::OrbImageMatcher, vo::ImageMatcher>(m, "OrbImageMatcher",
        "ORB-based image matcher (detect + match).")

        .def(nb::init<int, float>(),
        "max_features"_a = vo::default_max_features,
        "ratio_threshold"_a = vo::default_ratio_threshold,
        "Construct ORB matcher.\n\n"
        "Args:\n"
        "    max_features: Maximum features to detect (default: 2000).\n"
        "    ratio_threshold: Lowe's ratio test threshold (default: 0.75).");

    // ==================== create_matcher factory ====================
    m.def("create_matcher", [](std::string_view name) {
        auto matcher = vo::create_matcher(name);
        if (!matcher) {
            throw std::runtime_error("Unknown matcher: " + std::string(name));
        }
        return matcher;
    }, "name"_a,
    "Create a matcher by name.\n\n"
    "Args:\n"
    "    name: Matcher name - 'orb' (default) or 'matchanything'.\n\n"
    "Returns:\n"
    "    ImageMatcher instance.");

    // ==================== CameraIntrinsics ====================
    nb::class_<vo::CameraIntrinsics>(m, "CameraIntrinsics",
        "Camera intrinsic parameters.")

        .def(nb::init<>())

        .def_rw("fx", &vo::CameraIntrinsics::fx, "Focal length x.")
        .def_rw("fy", &vo::CameraIntrinsics::fy, "Focal length y.")
        .def_rw("cx", &vo::CameraIntrinsics::cx, "Principal point x.")
        .def_rw("cy", &vo::CameraIntrinsics::cy, "Principal point y.")

        .def_static("load_from_yaml", [](std::string_view filepath) {
            return unwrap_expected(vo::CameraIntrinsics::load_from_yaml(filepath));
        }, "filepath"_a,
        "Load intrinsics from YAML file.\n\n"
        "Args:\n"
        "    filepath: Path to YAML file.\n\n"
        "Returns:\n"
        "    CameraIntrinsics instance.")

        .def("__repr__", [](vo::CameraIntrinsics const& self) {
            return "CameraIntrinsics(fx=" + std::to_string(self.fx) +
                   ", fy=" + std::to_string(self.fy) +
                   ", cx=" + std::to_string(self.cx) +
                   ", cy=" + std::to_string(self.cy) + ")";
        });

    // ==================== MotionEstimate ====================
    nb::class_<vo::MotionEstimate>(m, "MotionEstimate",
        "Result of motion estimation between two frames.")

        .def_ro("rotation", &vo::MotionEstimate::rotation,
        "Rotation matrix (3x3 numpy array).")

        .def_ro("translation", &vo::MotionEstimate::translation,
        "Translation vector (3-element numpy array), unit norm.")

        .def_ro("inliers", &vo::MotionEstimate::inliers,
        "Number of RANSAC inliers.")

        .def_ro("valid", &vo::MotionEstimate::valid,
        "Whether estimation succeeded.")

        .def("__bool__", [](vo::MotionEstimate const& self) {
            return self.valid;
        });

    // ==================== MotionEstimator ====================
    nb::class_<vo::MotionEstimator>(m, "MotionEstimator",
        "Estimates camera motion from matched feature points.")

        .def(nb::init<vo::CameraIntrinsics const&, double, double>(),
        "intrinsics"_a,
        "ransac_threshold"_a = vo::default_ransac_threshold,
        "ransac_confidence"_a = vo::default_ransac_confidence,
        "Construct a new Motion Estimator.\n\n"
        "Args:\n"
        "    intrinsics: Camera intrinsic parameters.\n"
        "    ransac_threshold: RANSAC reprojection threshold in pixels (default: 1.0).\n"
        "    ransac_confidence: RANSAC confidence level 0-1 (default: 0.999).")

        .def("estimate", [](vo::MotionEstimator const& self,
                            std::vector<cv::Point2f> const& points1,
                            std::vector<cv::Point2f> const& points2) {
            return self.estimate(points1, points2);
        }, "points1"_a, "points2"_a,
        "Estimate motion from matched points.\n\n"
        "Args:\n"
        "    points1: Points in first image.\n"
        "    points2: Corresponding points in second image.\n\n"
        "Returns:\n"
        "    MotionEstimate containing R, t if successful.")

        .def_prop_ro("intrinsics", &vo::MotionEstimator::intrinsics,
        "Get the camera intrinsics.");

    // ==================== Pose ====================
    nb::class_<vo::Pose>(m, "Pose",
        "Represents an absolute camera pose in the world frame.")

        .def(nb::init<>())

        .def_rw("rotation", &vo::Pose::rotation,
        "Rotation matrix (3x3 numpy array).")

        .def_rw("translation", &vo::Pose::translation,
        "Translation vector (3-element numpy array).")

        .def_static("identity", &vo::Pose::identity,
        "Create identity pose (origin).")

        .def("compose", &vo::Pose::compose, "motion"_a,
        "Compose this pose with a relative transform.\n\n"
        "Args:\n"
        "    motion: Relative motion (MotionEstimate) from this pose.\n\n"
        "Returns:\n"
        "    New absolute pose after applying the transform.");

    // ==================== Trajectory ====================
    nb::class_<vo::Trajectory>(m, "Trajectory",
        "Accumulates camera poses to build a trajectory.")

        .def(nb::init<>(),
        "Construct empty trajectory starting at origin.")

        .def("add_motion", &vo::Trajectory::add_motion, "motion"_a,
        "Add a relative motion estimate to the trajectory.\n\n"
        "Args:\n"
        "    motion: Relative motion from current to next frame.\n\n"
        "Returns:\n"
        "    True if motion was valid and added, False otherwise.")

        .def_prop_ro("poses", &vo::Trajectory::poses,
        "Get all poses in the trajectory.")

        .def_prop_ro("current_pose", &vo::Trajectory::current_pose,
        "Get the current (latest) pose.")

        .def("size", &vo::Trajectory::size,
        "Get the number of poses in the trajectory.")

        .def("empty", &vo::Trajectory::empty,
        "Check if trajectory is empty (only has origin).")

        .def("reset", &vo::Trajectory::reset,
        "Reset trajectory to origin.")

        .def("save_to_json", [](vo::Trajectory const& self, std::string_view filepath) {
            unwrap_expected_void(self.save_to_json(std::filesystem::path(filepath)));
        }, "filepath"_a,
        "Save trajectory to JSON file.")

        .def("to_json", &vo::Trajectory::to_json,
        "Convert trajectory to JSON string.")

        .def("__len__", &vo::Trajectory::size)

        .def("__getitem__", [](vo::Trajectory const& self, size_t index) {
            if (index >= self.size()) {
                throw nb::index_error();
            }
            return self.poses()[index];
        }, "index"_a);
}
