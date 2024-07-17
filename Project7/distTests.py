import unittest
import numpy as np
from analysis import Analysis
from scipy.spatial.distance import cdist
import timeit
"""
Im writing code that calculates distances between sets of points in a dataset. 
I have a custom implementation of the Lp norm, that beats the default implementation in scipy.spatial.distance.cdist.
However, they beat me on normal norms.

"""


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
    for p in p_values:
        val_list.append(test_performance(a, b, Analysis.lp_norm_v2_pList, p))

    # cdist comparisons
    v1 = test_performance(a, b, lambda x, y: cdist(x, y, 'cityblock'))
    v2 = test_performance(a, b, lambda x, y: cdist(x, y, 'euclidean'))
    v3 = test_performance(a, b, lambda x, y: cdist(x, y, 'minkowski', p=3))
    v4 = test_performance(a, b, lambda x, y: cdist(x, y, 'minkowski', p=4.5))
    if np.inf in p_values:
        v5 = test_performance(a, b, lambda x, y: cdist(x, y, 'chebyshev'))
    for i, pz in enumerate(p_values):
        print(f"Custom L{pz} norm matches cdist: {np.allclose(val_list[i], [v1, v2, v3, v4, v5][i])}")
        print(f"Maximum difference: {np.max(np.abs(val_list[i] - [v1, v2, v3, v4, v5][i]))}")


# Generate a large random dataset
a, b = generate_random_data(500, 300, 1000)  # Example: 1000 samples, 100 dimensions

# Compare performance of custom norm functions against cdist
compare_with_cdist(a, b)