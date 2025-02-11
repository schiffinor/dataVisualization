import numpy as np
import scipy.spatial.distance as spd
import numpyProj as npP

def arrayPrepper(a, b, test=False):
    # np.abs(a[:, None, :] - b[None, :, :])
    return npP.compute_abs_difference(a, b)

def lp_norm_v2_pList(a: np.ndarray, b: np.ndarray, p: int | float = 2, debug: bool = True) -> np.ndarray:
    """
    This takes in two arrays and returns the Lp norm of the difference between the two arrays.
    :param a: ndarray.
        Input array 1
    :param b: ndarray.
        Input array 2
    :param p: float.
        The power of the norm.
    :param debug: bool.
        If True, print debug information and use custom slower methods.
    :return: ndarray.
        Lp norm of the difference between the two arrays.
    """
    if a.ndim != b.ndim:
        raise ValueError("Arrays must have the same number of dimensions")
    if a.ndim != 2:
        raise ValueError("Arrays must be 2D point lists")
    if a.shape[1] != b.shape[1] and a.shape[1] != b.T.shape[1]:
        raise ValueError("Arrays must have the same shape")
    if np.array_equal(a, b):
        return np.zeros(a.shape[:-1])
    if p < 0:
        raise ValueError("p must be greater than 0")
    if not isinstance(p, int) and not isinstance(p, float):
        print(type(p))
        raise ValueError("p must be a float or an integer")
    pfloat = float(p)
    inv_p = float(float(1) / pfloat) if p != 0 else np.inf
    if debug or not (p in {1, 2, 0, np.inf} or isinstance(p, int) or p.is_integer()):
        med_array = arrayPrepper(a, b)
        array_made = True
    else:
        med_array = np.array([0])
        array_made = False
    if p == np.inf:
        return spd.cdist(a, b, 'chebyshev') if not debug else np.max(med_array, axis=-1)
    elif p == 1:
        return spd.cdist(a, b, 'cityblock') if not debug else np.einsum("...i->...", med_array)
    elif p == 2:
        return spd.cdist(a, b, 'euclidean') if not debug else np.sqrt(np.einsum("...i,...i->...", med_array, med_array))
    elif p == 0:
        return spd.cdist(a, b, 'hamming') if not debug else np.einsum("...i->...", med_array > 0)
    elif isinstance(p, int) or p.is_integer():
        p = int(p)
        if not debug:
            if p == 1:
                return spd.cdist(a, b, 'cityblock')
            if p == 2:
                return spd.cdist(a, b, 'euclidean')
        ein_string = "...i," + ",".join(["...i" for _ in range(p - 1)]) + "->..."
        diff_copies = [med_array for _ in range(p)]
        if p == 2:
            return np.sqrt(np.einsum(ein_string, *diff_copies))
        return np.float_power(np.einsum(ein_string, *diff_copies), inv_p)
    elif not array_made:
        med_array = arrayPrepper(a, b)
    if p.as_integer_ratio()[0] == 1 and p.as_integer_ratio()[1] > 1:
        p_denom = p.as_integer_ratio()[1]
        ein_string = "...i," + ",".join(["...i" for _ in range(p_denom - 1)]) + "->..."
        post_med_array = np.float_power(med_array, p)
        diff_copies = [post_med_array for _ in range(p_denom)]
        return np.einsum(ein_string, *diff_copies)
    return np.float_power(np.einsum("...i->...", np.float_power(med_array, pfloat)), inv_p)