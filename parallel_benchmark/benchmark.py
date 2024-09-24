import time
import math
import multiprocessing

import pandas as pd
import matplotlib.pyplot as plt
import parallel_processing_cpp


def process_chunk_python(start, end):
    return sum(i % 1000 for i in range(start, end))  # Simple modulo operation to limit value size


def parallel_sum_python(n, num_processes):
    chunk_size = n // num_processes
    pool = multiprocessing.Pool(processes=num_processes)

    chunks = [
        (i * chunk_size, n if i == num_processes - 1 else (i + 1) * chunk_size)
        for i in range(num_processes)
    ]

    results = pool.starmap(process_chunk_python, chunks)
    return sum(results)


def benchmark(n_values, num_processes):
    results = []

    for n in n_values:
        print(f"Benchmarking with n = {n}")

        start = time.time()
        result_py = parallel_sum_python(n, num_processes)
        py_time = time.time() - start

        start = time.time()
        result_cpp = parallel_processing_cpp.parallel_sum_pthread(n, num_processes)
        cpp_pthread_time = time.time() - start

        speedup_pthread = py_time / cpp_pthread_time

        results.append(
            {
                "n": n,
                "Python Time (s)": py_time,
                "C++ Time (s)": cpp_pthread_time,
                "Speedup (Python/C++)": speedup_pthread,
            }
        )

        print(result_py, result_cpp)
        assert result_py == result_cpp

    return pd.DataFrame(results)


def plot_results(df, file_name="parallel_processing_benchmark.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(df["n"], df["Python Time (s)"], label="Python", marker="o")
    plt.plot(df["n"], df["C++ Time (s)"], label="C++ (pthreads)", marker="o")

    plt.xlabel("n (Data Size)")
    plt.ylabel("Execution Time (s)")
    plt.title("Parallel Processing Benchmark: Python vs C++ (pthreads)")
    plt.legend()
    plt.grid(True)

    plt.savefig(file_name)
    plt.show()


def main():
    n_values = [
        10_000,
        50_000,
        100_000,
        200_000,
        500_000,
        1_000_000,
        5_000_000,
        10_000_000,
        50_000_000,
        100_000_000,
        300_000_000,
        500_000_000,
        1_000_000_000,
    ]
    num_processes = 5
    df = benchmark(n_values, num_processes)
    print(df)
    df.to_csv("parallel_processing_benchmark_results.csv", index=False)

    plot_results(df)


if __name__ == "__main__":
    main()


"""
             n  Python Time (s)  C++ Time (s)  Speedup (Python/C++)
0        10000         0.335266      0.000095           3533.185930
1        50000         0.331780      0.000079           4191.527108
2       100000         0.335085      0.000097           3453.189189
3       200000         0.333086      0.000121           2750.125984
4       500000         0.335471      0.000164           2045.156977
5      1000000         0.341201      0.000158           2155.271084
6      5000000         0.368595      0.000450            818.855403
7     10000000         0.396034      0.000817            484.564177
8     50000000         0.666497      0.003779            176.371672
9    100000000         1.004328      0.007450            134.807252
10   300000000         2.339031      0.022133            105.680168
11   500000000         3.692330      0.036883            100.109601
12  1000000000         7.016980      0.073660             95.261876
"""
