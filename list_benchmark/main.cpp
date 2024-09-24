#include <pybind11/pybind11.h>

#include <numeric>
#include <vector>

namespace py = pybind11;

long long large_list_sum(int size) {
  std::vector<int> large_list(size);
  std::iota(large_list.begin(), large_list.end(), 0);
  return std::accumulate(large_list.begin(), large_list.end(), 0LL);
}

PYBIND11_MODULE(list_processing_cpp, m) {
  m.def("large_list_sum", &large_list_sum);
}

// > $CXX -O3 -Wall -shared -std=c++17 -undefined dynamic_lookup -fPIC $(poetry
// run python3 -m pybind11 --includes) main.cpp -o
// list_processing_cpp$(python3-config --extension-suffix) -fopenmp