import math
import unittest
import numpy as np
from analysis import Analysis
lp_norm_v2_pList = Analysis.lp_norm_v2_pList
from scipy.spatial.distance import cdist
import timeit


def generate_random_data(size1, size2, dimensions):
    """Generate a random dataset of specified size and dimensions."""
    return np.random.rand(size1, dimensions), np.random.rand(size2, dimensions)


def test_performance(a, b, func, p=None):
    """Measure execution time of a given function."""
    if p is not None:
        stmt = lambda: func(a, b, p)
    else:
        stmt = lambda: func(a, b)
    time = timeit.timeit(stmt, number=10)
    print(f"{func.__name__ if p is None else func.__name__ + ' with p=' + str(p)}: {time:.6f} seconds")
    return stmt()


def compare_with_cdist(a, b, p_values=None):
    """Compare custom norm functions against cdist for L1, L2, and L-infinity norms."""
    # Custom implementations
    if p_values is None:
        p_values = [1, 2, 3, 4.5, np.inf]
    val_list = []
    c_vals = []
    for p in p_values:
        val_list.append(test_performance(a, b, lp_norm_v2_pList, p))
        if p == np.inf:
            c_vals.append(test_performance(a, b, lambda x, y: cdist(a, b, "chebyshev")))
        elif p == 0:
            c_vals.append(test_performance(a, b, lambda x, y: cdist(a, b, "hamming")))
        elif p == 1:
            c_vals.append(test_performance(a, b, lambda x, y: cdist(a, b, "cityblock")))
        elif p == 2:
            c_vals.append(test_performance(a, b, lambda x, y: cdist(a, b, "euclidean")))
        else:
            c_vals.append(test_performance(a, b, lambda x, y: cdist(a, b, "minkowski", p=p)))

    for i, pz in enumerate(p_values):
        print(f"Custom L{pz} norm matches cdist: {np.allclose(val_list[i], c_vals[i])}")
        print(f"Maximum difference: {np.max(np.abs(val_list[i] - c_vals[i]))}")


# Generate a large random dataset
a, b = generate_random_data(500, 300, 100)  # Example: 1000 samples, 100 dimensions

# Compare performance of custom norm functions against cdist
compare_with_cdist(a, b, [0, 1, 2, 3, 4.5, math.pi, np.inf])
