import time
import pandas as pd
import matplotlib.pyplot as plt
import list_processing_cpp


def large_list_sum(size):
    large_list = [i for i in range(size)]
    return sum(large_list)


def benchmark(data_sizes):
    results = []

    for size in data_sizes:
        print(f"Benchmarking for size: {size}")

        # Python version
        start = time.time()
        py_result = large_list_sum(size)
        py_time = time.time() - start

        # C++ version
        start = time.time()
        cpp_result = list_processing_cpp.large_list_sum(size)
        cpp_time = time.time() - start

        speedup = py_time / cpp_time

        print(py_result, cpp_result)
        assert py_result == cpp_result

        results.append(
            {
                "Size": size,
                "Python Time (s)": py_time,
                "C++ Time (s)": cpp_time,
                "Speedup (Python/C++)": speedup,
            }
        )

        print(f"Python time: {py_time:.4f} seconds")
        print(f"C++ time: {cpp_time:.4f} seconds")
        print(f"C++ is {speedup:.2f}x faster")

    return pd.DataFrame(results)


def plot_results(df, file_name="speedup_comparison.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(df["Size"], df["Speedup (Python/C++)"], label="Speedup (Python/C++)", marker="o")
    plt.xlabel("Input Size (Number of Elements)")
    plt.ylabel("Speedup (Python/C++)")
    plt.title("Python vs C++ Speedup Comparison")
    plt.xscale("log")
    plt.grid(True)
    plt.legend()

    plt.savefig(file_name)
    plt.show()


def main():
    data_sizes = [10**5, 10**6, 10**7, 10**8, 2 * 10**8, 4 * 10**8, 8 * 10**8, 10**9]
    df = benchmark(data_sizes)
    print(df)
    df.to_csv("speedup_comparison_results.csv", index=False)
    plot_results(df)


if __name__ == "__main__":
    main()


"""
         Size  Python Time (s)  C++ Time (s)  Speedup (Python/C++)
0      100000         0.007802      0.000112             69.771855
1     1000000         0.035118      0.000477             73.648000
2    10000000         0.174350      0.003189             54.670754
3   100000000         1.714007      0.037011             46.310553
4   200000000         3.347352      0.071111             47.072235
5   400000000         6.842253      0.142053             48.167022
6   800000000        20.027947      0.244773             81.822483
7  1000000000        25.969393      0.308420             84.201407
"""
