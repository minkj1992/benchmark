#include <omp.h>
#include <pybind11/pybind11.h>

#include <numeric>
#include <thread>
#include <vector>

namespace py = pybind11;

void process_chunk(long long start, long long end, long long& result) {
  result = 0;
  for (long long i = start; i < end; ++i) {
    result += i % 1000;  // Simple modulo operation to limit value size
  }
}

long long parallel_sum_pthread(long long n, int num_threads) {
  std::vector<std::thread> threads;
  std::vector<long long> results(num_threads);

  long long chunk_size = n / num_threads;

  for (int i = 0; i < num_threads; ++i) {
    long long start = i * chunk_size;
    long long end = (i == num_threads - 1) ? n : (i + 1) * chunk_size;
    threads.emplace_back(process_chunk, start, end, std::ref(results[i]));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return std::accumulate(results.begin(), results.end(), 0LL);
}

PYBIND11_MODULE(parallel_processing_cpp, m) {
  m.def("parallel_sum_pthread", &parallel_sum_pthread,
        "Perform parallel sum using pthread");
}

// $CXX - O3 - Wall - shared - std =
//     c++ 17 - undefined dynamic_lookup -
//     fPIC $(poetry run python3 - m pybind11-- includes)
//     parallel_processing.cpp - o parallel_processing_cpp$(python3 - config--
//     extension - suffix) - fopenmp