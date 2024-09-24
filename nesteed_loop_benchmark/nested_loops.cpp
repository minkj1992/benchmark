#include <pybind11/pybind11.h>

namespace py = pybind11;

long long nested_loops(int n) {
  long long result = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        result += i * j * k;
      }
    }
  }
  return result;
}

PYBIND11_MODULE(nested_loops_cpp, m) {
  m.def("nested_loops", &nested_loops, "Perform nested loops calculation");
}
