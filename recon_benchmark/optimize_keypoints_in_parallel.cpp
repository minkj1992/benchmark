#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace py = pybind11;

struct CameraParams {
  std::vector<double> K, R, t;
};

struct Detection {
  CameraParams cam_params;
  std::vector<double> loc;
  double confidence;
};

double mean_reprojection_error(
    const std::vector<double>& X,
    const std::map<std::string, Detection>& detections) {
  double total_error = 0.0;
  for (const auto& [_, detection] : detections) {
    double error = std::sqrt(std::pow(X[0] - detection.loc[0], 2) +
                             std::pow(X[1] - detection.loc[1], 2));
    total_error += error;
  }
  return total_error / detections.size();
}

std::vector<double> triangulate_from_observations(
    const std::map<std::string, Detection>& detections) {
  std::vector<double> X(3, 0.0);
  for (const auto& [_, detection] : detections) {
    for (int i = 0; i < 3; ++i) {
      X[i] += detection.loc[i % 2];
    }
  }
  for (int i = 0; i < 3; ++i) {
    X[i] /= detections.size();
  }
  return X;
}

std::vector<double> optimize_keypoint_routine_cpp(
    const std::map<std::string, Detection>& detections,
    double detection_confidence_threshold, int niter_nonlinear) {
  std::vector<double> X = triangulate_from_observations(detections);

  for (int iter = 0; iter < niter_nonlinear; ++iter) {
    double error = mean_reprojection_error(X, detections);
    if (error < detection_confidence_threshold) {
      break;
    }
    // Simulated optimization step
    for (int i = 0; i < 3; ++i) {
      X[i] += (std::rand() / static_cast<double>(RAND_MAX) - 0.5) * 0.01;
    }
  }

  return X;
}

std::vector<std::vector<double>> optimize_keypoints_in_parallel_cpp(
    const std::vector<std::map<std::string, Detection>>& people3d,
    int num_threads, double detection_confidence_threshold,
    int niter_nonlinear) {
  std::vector<std::vector<double>> optimized_keypoints(people3d.size());

#pragma omp parallel for num_threads(num_threads)
  for (size_t i = 0; i < people3d.size(); ++i) {
    optimized_keypoints[i] = optimize_keypoint_routine_cpp(
        people3d[i], detection_confidence_threshold, niter_nonlinear);
  }

  return optimized_keypoints;
}

PYBIND11_MODULE(optimize_cpp, m) {
  py::class_<CameraParams>(m, "CameraParams")
      .def(py::init<>())
      .def_readwrite("K", &CameraParams::K)
      .def_readwrite("R", &CameraParams::R)
      .def_readwrite("t", &CameraParams::t);

  py::class_<Detection>(m, "Detection")
      .def(py::init<>())
      .def_readwrite("cam_params", &Detection::cam_params)
      .def_readwrite("loc", &Detection::loc)
      .def_readwrite("confidence", &Detection::confidence);

  m.def("optimize_keypoints_in_parallel_cpp",
        &optimize_keypoints_in_parallel_cpp,
        "Optimized keypoint routine with OpenMP");
}