import time
import nested_loops_cpp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def nested_loops_python(n):
    result = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result += i * j * k
    return result


def nested_loops_numpy(n):
    arr = np.arange(n)
    return np.sum(arr[:, None, None] * arr[None, :, None] * arr[None, None, :])


def benchmark(n_values):
    results = []

    for n in n_values:
        # Python benchmark
        start = time.time()
        result_py = nested_loops_python(n)
        py_time = time.time() - start

        # C++ benchmark
        start = time.time()
        result_cpp = nested_loops_cpp.nested_loops(n)
        cpp_time = time.time() - start
        assert result_py == result_cpp

        # Numpy benchmark
        start = time.time()
        result_np = nested_loops_numpy(n)
        np_time = time.time() - start
        assert result_py == result_np

        results.append(
            {
                "n": n,
                "Python Time (s)": py_time,
                "C++ Time (s)": cpp_time,
                "Numpy Time (s)": np_time,
                "Speedup (Python/C++)": py_time / cpp_time,
                "Speedup (Python/Numpy)": py_time / np_time,
            }
        )

        print(
            f"n = {n} | Python time: {py_time:.4f}s | C++ time: {cpp_time:.4f}s | Numpy time: {np_time:.4f}s"
        )

    return pd.DataFrame(results)


def plot_results(df, file_name="performance_comparison.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(df["n"], df["Python Time (s)"], label="Python", marker="o")
    plt.plot(df["n"], df["C++ Time (s)"], label="C++", marker="o")
    plt.plot(df["n"], df["Numpy Time (s)"], label="Numpy", marker="o")
    plt.xlabel("n (Input Size)")
    plt.ylabel("Execution Time (s)")
    plt.title("Performance Comparison: Python vs C++ vs Numpy (Nested Loops)")
    plt.legend()
    plt.grid(True)

    plt.savefig(file_name)
    plt.show()


def main():
    n_values = [100, 200, 300, 400, 500, 600, 700, 1000, 2000]
    df = benchmark(n_values)
    print(df)
    df.to_csv("performance_results.csv", index=False)
    plot_results(df)


if __name__ == "__main__":
    main()


"""
      n  Python Time (s)  C++ Time (s)  Numpy Time (s)  Speedup (Python/C++)  Speedup (Python/Numpy)  
0   100         0.035146      0.000006        0.001146           5896.480000               30.666112   
1   200         0.266048      0.000038        0.009769           7018.150943               27.234002   
2   300         1.020466      0.000038        0.031718          26750.900000               32.173067   
3   400         2.300614      0.000051        0.079230          45302.704225               29.037136   
4   500         4.637607      0.000101        0.152984          45768.312941               30.314345   
5   600         8.101100      0.000128        0.295594          63274.629423               27.406174   
6   700        12.665226      0.000149        0.442128          84994.892800               28.646065   
7  1000        38.596019      0.000282        1.298317         136725.875000               29.727732   
8  2000       311.207247      0.001077      147.909453         288974.496569                2.104039
"""
