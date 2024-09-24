import time
from matplotlib import pyplot as plt
import numpy as np
import multiprocessing as mp

import pandas as pd
import optimize_cpp  # C++


def generate_people3d(num_people, num_keypoints):
    """Generate people data in Python format"""
    people3d = []
    for _ in range(num_people):
        detections = {}
        for keypoint in range(num_keypoints):
            detections[str(keypoint)] = {
                "loc": np.random.rand(2).tolist(),
                "confidence": np.random.rand(),
                "cam_params": {
                    "K": np.random.rand(3, 3).flatten().tolist(),
                    "R": np.random.rand(3, 3).flatten().tolist(),
                    "t": np.random.rand(3).tolist(),
                },
            }
        people3d.append(detections)
    return people3d


def convert_to_cpp_format(people3d_python):
    """Convert people data from Python to C++ format."""

    def convert_detection_to_cpp(detection):
        """Convert a single detection from Python to C++ format."""
        cpp_detection = optimize_cpp.Detection()
        cpp_detection.loc = detection["loc"]
        cpp_detection.confidence = detection["confidence"]

        cam_params = optimize_cpp.CameraParams()
        cam_params.K = detection["cam_params"]["K"]
        cam_params.R = detection["cam_params"]["R"]
        cam_params.t = detection["cam_params"]["t"]
        cpp_detection.cam_params = cam_params

        return cpp_detection

    people3d_cpp = []
    for person in people3d_python:
        cpp_detections = {
            keypoint_label: convert_detection_to_cpp(detection)
            for keypoint_label, detection in person.items()
        }
        people3d_cpp.append(cpp_detections)
    return people3d_cpp


def triangulate_from_observations(detections):
    """Simulates triangulation from observations to estimate the initial 3D position."""
    X = np.zeros(3)
    for detection in detections.values():
        for i in range(3):
            X[i] += detection["loc"][i % 2]  # Summing x, y coordinates
    X /= len(detections)  # Averaging to get the initial estimate
    return X


def mean_reprojection_error(X, detections):
    """Calculates the mean reprojection error between 3D keypoint and 2D detections."""
    total_error = 0.0
    for detection in detections.values():
        error = np.sqrt(np.sum((X[:2] - detection["loc"]) ** 2))  # Error in 2D space
        total_error += error
    return total_error / len(detections)


def optimize_keypoint_routine_python(detections, detection_confidence_threshold, niter_nonlinear):
    """Keypoint optimization routine for Python."""
    X = triangulate_from_observations(detections)  # Initialize X based on triangulation
    for _ in range(niter_nonlinear):
        error = mean_reprojection_error(X, detections)
        if error < detection_confidence_threshold:
            break
        # Update X with a small random step (as in C++ code)
        X += (np.random.rand(3) - 0.5) * 0.01
    return X


def optimize_keypoints_in_parallel_python(
    people3d, num_threads, detection_confidence_threshold, niter_nonlinear
):
    """Run keypoint optimization in parallel using Python."""
    with mp.Pool(num_threads) as pool:
        return pool.starmap(
            optimize_keypoint_routine_python,
            [(person, detection_confidence_threshold, niter_nonlinear) for person in people3d],
        )


def benchmark(
    num_people_list,
    num_keypoints,
    num_threads_list,
    detection_confidence_threshold,
    niter_nonlinear,
):
    """Run benchmark tests for Python and C++ implementations."""
    results = []

    for num_people in num_people_list:
        for num_threads in num_threads_list:
            people3d_python = generate_people3d(num_people, num_keypoints)

            # Python version benchmark
            start_time = time.time()
            optimized_python = optimize_keypoints_in_parallel_python(
                people3d_python, num_threads, detection_confidence_threshold, niter_nonlinear
            )
            python_time = time.time() - start_time

            # C++ version benchmark
            people3d_cpp = convert_to_cpp_format(people3d_python)
            start_time = time.time()
            optimized_cpp = optimize_cpp.optimize_keypoints_in_parallel_cpp(
                people3d_cpp, num_threads, detection_confidence_threshold, niter_nonlinear
            )
            cpp_time = time.time() - start_time
            speedup = python_time / cpp_time

            # Store results
            results.append(
                {
                    "num_people": num_people,
                    "num_threads": num_threads,
                    "python_time": python_time,
                    "cpp_time": cpp_time,
                    "speedup": speedup,
                }
            )

    return results


def plot_benchmark(df, num_threads_list):
    plt.figure(figsize=(10, 6))

    for num_threads in num_threads_list:
        subset = df[df["num_threads"] == num_threads]

        # Plot Python vs C++ for each num_threads
        plt.plot(
            subset["num_people"],
            subset["python_time"],
            marker="o",
            label=f"Python ({num_threads} threads)",
        )
        plt.plot(
            subset["num_people"],
            subset["cpp_time"],
            marker="o",
            label=f"C++ ({num_threads} threads)",
            linestyle="--",
        )

    plt.title("Python vs C++ Execution Time Comparison by Number of Threads")
    plt.xlabel("Number of People")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    num_people_list = [100, 500, 1000, 2000]  # Varying number of people
    num_keypoints = 17
    num_threads_list = [2, 4, 8]  # Varying number of threads
    detection_confidence_threshold = 0.1
    niter_nonlinear = 100

    # Run benchmark
    results = benchmark(
        num_people_list,
        num_keypoints,
        num_threads_list,
        detection_confidence_threshold,
        niter_nonlinear,
    )

    # Create DataFrame from results
    df = pd.DataFrame(results)
    print(df)

    # Plot the benchmark results
    plot_benchmark(df, num_threads_list)
