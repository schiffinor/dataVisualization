"""analysis.py
Run statistical analyses and plot Numpy ndarray data
YOUR NAME HERE
CS 251/2: Data Analysis and Visualization
Spring 2024
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as spd
import numpyProjBuilt.numpyProj as npP
from dataClass import *




def show():
    """Simple wrapper function for matplotlib's show function.

    (Does not require modification)
    """
    plt.show()


def arrayPrepper(a, b, test=False):
    # np.abs(a[:, None, :] - b[None, :, :])
    return npP.compute_abs_difference(a, b)


class Analysis:
    def __init__(self, data):
        """

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        """
        self.data = data

        # Make plot font sizes legible
        plt.rnparams.update({'font.size': 18})

    @staticmethod
    def l1_norm(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape and a.shape != b.T.shape:
            raise ValueError("Arrays must have the same shape")
        if a.ndim != 1:
            raise ValueError("Arrays must be 1D")
        if np.array_equal(a, b):
            return 0
        return np.einsum("i->", np.abs(b - a))

    @staticmethod
    def l1_norm_ndim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        This takes in two arrays and returns the L1 norm of the difference between the two arrays.
        Since L1 norm is a vector norm it will return an n-1 dimensional array where n is the number of dimensions
        of the input arrays. The last dimension will be reduced to a scalar value.
        That is for a 3D array of shape (3, 2, 4) the output will be of shape (3, 2).
        :param a: ndarray.
            Input array 1
        :param b: ndarray.
            Input array 2
        :return: ndarray.
            L1 norm of the difference between the two arrays.
        """
        if a.shape != b.shape and a.shape != b.T.shape:
            raise ValueError("Arrays must have the same shape")
        if np.array_equal(a, b):
            return np.zeros(a.shape[:-1])
        return np.einsum("...i->...", np.abs(b - a))

    @staticmethod
    def l2_norm(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape and a.shape != b.T.shape:
            raise ValueError("Arrays must have the same shape")
        if a.ndim != 1:
            raise ValueError("Arrays must be 1D")
        if np.array_equal(a, b):
            return 0
        c = np.abs(b - a)
        return np.sqrt(np.einsum('i,i->', c, c))

    @staticmethod
    def l2_norm_ndim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        This takes in two arrays and returns the L2 norm of the difference between the two arrays.
        Since L2 norm is a vector norm it will return an n-1 dimensional array where n is the number of dimensions
        of the input arrays. The last dimension will be reduced to a scalar value.
        That is for a 3D array of shape (3, 2, 4) the output will be of shape (3, 2).
        :param a: ndarray.
            Input array 1
        :param b: ndarray.
            Input array 2
        :return: ndarray.
            L2 norm of the difference between the two arrays.
        """
        if a.shape != b.shape and a.shape != b.T.shape:
            raise ValueError("Arrays must have the same shape")
        if np.array_equal(a, b):
            return np.zeros(a.shape[:-1])
        c = np.abs(b - a)
        return np.sqrt(np.einsum('...i,...i->...', c, c))

    @staticmethod
    def lp_norm(a: np.ndarray, b: np.ndarray, p: float = 2) -> float:
        if a.shape != b.shape and a.shape != b.T.shape:
            raise ValueError("Arrays must have the same shape")
        if a.ndim != 1:
            raise ValueError("Arrays must be 1D")
        if np.array_equal(a, b):
            return 0
        if p == 1:
            return Analysis.l1_norm(a, b)
        if p == 2:
            return Analysis.l2_norm(a, b)
        if p == np.inf:
            return Analysis.l_inf_norm(a, b)
        if p < 1 or not p.is_integer():
            return Analysis.lp_norm_alt(a, b, p)  # This is the only place where the alternative method is called
        p = int(p)
        if p < 0:
            raise ValueError("p must be greater than 0")
        if not isinstance(p, int):
            raise ValueError("p must be an integer")
        pfloat = float(p)
        inv_p = float(float(1) / pfloat)
        ein_string = "i," + ",".join(["i" for _ in range(p - 1)]) + "->"
        c = np.abs(b - a)
        diff_copies = [c for _ in range(p)]
        return np.float_power(np.einsum(ein_string, *diff_copies), inv_p)

    @staticmethod
    def lp_norm_ndim(a: np.ndarray, b: np.ndarray, p: float = 2) -> np.ndarray:
        """
        This takes in two arrays and returns the Lp norm of the difference between the two arrays.
        Since Lp norm is a vector norm it will return an n-1 dimensional array where n is the number of dimensions
        of the input arrays. The last dimension will be reduced to a scalar value.
        That is for a 3D array of shape (3, 2, 4) the output will be of shape (3, 2).
        :param a: ndarray.
            Input array 1
        :param b: ndarray.
            Input array 2
        :param p: float.
            The power of the norm.
        :return: ndarray.
            Lp norm of the difference between the two arrays.
        """
        if a.shape != b.shape and a.shape != b.T.shape:
            raise ValueError("Arrays must have the same shape")
        if np.array_equal(a, b):
            return np.zeros(a.shape[:-1])
        if p == 1:
            return Analysis.l1_norm_ndim(a, b)
        if p == 2:
            return Analysis.l2_norm_ndim(a, b)
        if p == np.inf:
            return Analysis.l_inf_norm_ndim(a, b)
        if p < 1 or not p.is_integer():
            return Analysis.lp_norm_alt_ndim(a, b, p)  # This is the only place where the alternative method is called
        p = int(p)
        if p < 0:
            raise ValueError("p must be greater than 0")
        if not isinstance(p, int):
            raise ValueError("p must be an integer")
        pfloat = float(p)
        inv_p = float(float(1) / pfloat)
        ein_string = "...i," + ",".join(["...i" for _ in range(p - 1)]) + "->..."
        c = np.abs(b - a)
        diff_copies = [c for _ in range(p)]
        return np.float_power(np.einsum(ein_string, *diff_copies), inv_p)

    @staticmethod
    def lp_norm_alt(a: np.ndarray, b: np.ndarray, p: float = 2) -> float:
        """
        This takes in two arrays and returns the Lp norm of the difference between the two arrays.
        This is a recursive function that will call itself until the p value is less than 1.
        this is specifically for calculating the Lp norm when p is less than 1. It is less efficient.

        :param a: ndarray.
            Input array 1
        :param b: ndarray.
            Input array 2
        :param p: float.
            The power of the norm.
        :return: float.
            Lp norm of the difference between the two arrays.
        """
        if not isinstance(p, float):
            warnings.warn("p is not a float, it will be converted to a float")
        if p > 1:
            return Analysis.lp_norm_alt(b, a, 1 / p)
        return np.float_power(np.sum(np.abs(b - a) ** p), 1 / p)

    @staticmethod
    def lp_norm_alt_ndim(a: np.ndarray, b: np.ndarray, p: float = 2) -> np.ndarray:
        """
        This takes in two arrays and returns the Lp norm of the difference between the two arrays.
        This is a recursive function that will call itself until the p value is less than 1.
        this is specifically for calculating the Lp norm when p is less than 1. It is less efficient.

        :param a: ndarray.
            Input array 1
        :param b: ndarray.
            Input array 2
        :param p: float.
            The power of the norm.
        :return: ndarray.
            Lp norm of the difference between the two arrays.
        """
        if not isinstance(p, float):
            warnings.warn("p is not a float, it will be converted to a float")
        if p > 1:
            return Analysis.lp_norm_alt_ndim(b, a, 1 / p)
        if np.array_equal(a, b):
            return np.zeros(a.shape[:-1])
        return np.float_power(np.sum(np.abs(b - a) ** p, axis=-1), 1 / p)


    @staticmethod
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


    @staticmethod
    def l_inf_norm(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape and a.shape != b.T.shape:
            raise ValueError("Arrays must have the same shape")
        if a.ndim != 1:
            raise ValueError("Arrays must be 1D")
        if np.array_equal(a, b):
            return 0
        return np.max(np.abs(b - a), axis=0)

    @staticmethod
    def l_inf_norm_ndim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        This takes in two arrays and returns the L infinity norm of the difference between the two arrays.
        Since L infinity norm is a vector norm it will return an n-1 dimensional array where n is the number of dimensions
        of the input arrays. The last dimension will be reduced to a scalar value.
        That is for a 3D array of shape (3, 2, 4) the output will be of shape (3, 2).
        :param a: ndarray.
            Input array 1
        :param b: ndarray.
            Input array 2
        :return: ndarray.
            L infinity norm of the difference between the two arrays.
        """
        if a.shape != b.shape and a.shape != b.T.shape:
            raise ValueError("Arrays must have the same shape")
        if np.array_equal(a, b):
            return np.zeros(a.shape[:-1])
        return np.max(np.abs(b - a), axis=-1)

    @staticmethod
    def l0_norm(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape and a.shape != b.T.shape:
            raise ValueError("Arrays must have the same shape")
        if a.ndim != 1:
            raise ValueError("Arrays must be 1D")
        if np.array_equal(a, b):
            return 0
        return float(np.einsum("i->", np.abs(b - a) > 0))

    @staticmethod
    def l0_norm_ndim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        This takes in two arrays and returns the L0 norm of the difference between the two arrays.
        Since L0 norm is a vector norm it will return an n-1 dimensional array where n is the number of dimensions
        of the input arrays. The last dimension will be reduced to a scalar value.
        That is for a 3D array of shape (3, 2, 4) the output will be of shape (3, 2).
        :param a: ndarray.
            Input array 1
        :param b: ndarray.
            Input array 2
        :return: ndarray.
            L0 norm of the difference between the two arrays.
        """
        if a.shape != b.shape and a.shape != b.T.shape:
            raise ValueError("Arrays must have the same shape")
        if np.array_equal(a, b):
            return np.zeros(a.shape[:-1])
        return np.einsum("...i->...", np.abs(b - a) > 0)

    def set_data(self, data):
        """Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        """
        self.data = data

    def min(self, headers, rows=None):
        """Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        """
        data_selection = self.data.select_data(headers, rows)
        return np.min(data_selection, axis=0)

    def max(self, headers, rows=None):
        """Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        """
        data_selection = self.data.select_data(headers, rows)
        return np.max(data_selection, axis=0)

    def range(self, headers, rows=None):
        """Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        """
        data_selection = self.data.select_data(headers, rows)
        return [np.min(data_selection, axis=0), np.max(data_selection, axis=0)]

    def mean(self, headers, rows=None):
        """Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: There should be no loops in this method!
        """
        data_selection = self.data.select_data(headers, rows)
        height = data_selection.shape[0]
        return (1 / height) * np.sum(data_selection, axis=0)

    def var(self, headers, rows=None):
        """Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE:
        - You CANNOT use np.var or np.mean here!
        - There should be no loops in this method!
        """
        data_selection = self.data.select_data(headers, rows)
        height = data_selection.shape[0]
        means = self.mean(headers, rows)
        return (1 / (height - 1)) * np.sum((data_selection - means) ** 2, axis=0)

    def std(self, headers, rows=None):
        """Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over, or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE:
        - You CANNOT use np.var, np.std, or np.mean here!
        - There should be no loops in this method!
        """
        return np.sqrt(self.var(headers, rows))

    @staticmethod
    def show():
        """Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        """
        plt.show()

    def scatter(self, ind_var, dep_var, title):
        """Creates a simple scatter plot with "x" variable in the dataset `ind_var` and "y" variable in the dataset
        `dep_var`. Both `ind_var` and `dep_var` should be strings in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        """
        ind_data = self.data.select_data([ind_var])
        dep_data = self.data.select_data([dep_var])
        plt.scatter(ind_data, dep_data)
        plt.title(title)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        return ind_data, dep_data

    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        """Create a pair plot: grid of scatter plots showing all combinations of variables in `data_vars` in the
        x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        1. Make the len(data_vars) x len(data_vars) grid of scatterplots
        2. The y axis of the FIRST column should be labeled with the appropriate variable being plotted there.
        The x axis of the LAST row should be labeled with the appropriate variable being plotted there.
        3. Only label the axes and ticks on the FIRST column and LAST row. There should be no labels on other plots
        (it looks too cluttered otherwise!).
        4. Do have tick MARKS on all plots (just not the labels).
        5. Because variables may have different ranges, your pair plot should share the y axis within columns and
        share the x axis within rows. To implement this, add
            sharex='col', sharey='row'
        to your plt.subplots call.

        NOTE: For loops are allowed here!
        """
        size = len(data_vars)
        figure, ax = plt.subplots(size, size, figsize=fig_sz, sharex='col', sharey='row')
        for row, dep in enumerate(data_vars):
            for col, ind in enumerate(data_vars):
                ind_data = self.data.select_data([ind])
                dep_data = self.data.select_data([dep])
                place = ax[row, col]
                place.scatter(ind_data, dep_data)
                if row == size - 1:
                    place.set_xlabel(ind)
                if col == 0:
                    place.set_ylabel(dep)
        plt.title(title)
        return figure, ax

    def l_centrality(self, headers, rows=None, metric=l_inf_norm):
        """Computes the L-infinity centrality for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of L-infinity centrality over, or over all indices if rows=[]

        Returns
        -----------
        l_inf: ndarray. shape=(len(headers),)
            L-infinity centrality values computed considering all of the selected header variables

        NOTE: There should be no loops in this method!
        """
        if metric == self.l1_norm:
            norm = 1
        elif metric == self.l2_norm:
            norm = 2
        elif metric == self.lp_norm:
            raise ValueError("Lp norm not supported")
        elif metric == self.l_inf_norm:
            norm = "inf"
        else:
            raise ValueError("Invalid metric")

        data_selection = self.data.select_data(headers, rows).copy()
        init_size = data_selection.shape[0]
        rows = []
        for i in range(init_size):

            row = data_selection[0]
            tiler = np.tile(row, (data_selection.shape[0], 1))
            tensors = []
            if norm == 1:
                ein_string = 'ijk->jk'
                tensor1 = np.array([tiler, -data_selection])
                tensors.append(tensor1)
            elif norm == 2:
                ein_string = 'ijk,ijk->j'
                tensor1 = np.array([tiler, data_selection, -2 * tiler])
                tensor2 = np.array([tiler, data_selection, data_selection])
                tensors.append(tensor1)
                tensors.append(tensor2)
            elif norm == "inf":
                ein_string = 'ijk->jk'
                tensor1 = np.array([tiler, -data_selection])
                tensors.append(tensor1)
            else:
                raise ValueError("Invalid norm")
            prod_vec = np.einsum(ein_string, *tensors)
            if norm == "inf":
                prod_vec = np.max(np.abs(prod_vec), axis=1)
            elif norm == 1:
                prod_vec = np.einsum('jk->j', np.abs(prod_vec))
            data_selection = data_selection[1:]
            prod_vec = np.pad(prod_vec, pad_width=(i, 0))
            rows.append(prod_vec.tolist())

        np.set_printoptions(formatter={'float': '{:.2f}'.format})
        matrix = np.einsum('ijk,ikj->jk', np.array([np.ones((len(rows), len(rows))), rows]),
                           np.array([rows, np.ones((len(rows), len(rows)))]))
        return np.sqrt(np.max(matrix, axis=0)) if norm == 2 else np.max(matrix, axis=0)

    def l_centrality_alt(self, headers, rows=None, metric=l_inf_norm):
        """Computes the L-infinity centrality for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of L-infinity centrality over, or over all indices if rows=[]

        Returns
        -----------
        l_inf: ndarray. shape=(len(headers),)
            L-infinity centrality values computed considering all of the selected header variables

        NOTE: There should be no loops in this method!
        """
        norm = self.l_inf_norm
        if metric == self.l1_norm:
            norm = self.l1_norm
        elif metric == self.l2_norm:
            norm = self.l2_norm

        data_selection = self.data.select_data(headers, rows)
        point_count = data_selection.shape[0]
        distance_array = np.ndarray(shape=point_count)
        # Calculates order of the row count for the dataset, ie floor of the base 10 log of the row count.
        order = math.floor(math.log10(point_count))
        for i in range(point_count):
            if i % (100 * (math.pow(10, order - 5))) == 0 and point_count >= 10000:
                ratio = i / point_count
                print("String output {:.2%}".format(ratio))
            new_max = np.max(np.array([norm(data_selection[i], data_selection[j]) for j in range(point_count)]))
            distance_array[i] = new_max
        return distance_array

    def l_centroid(self, headers, rows=None, metric=l_inf_norm):
        """Computes the L-infinity centrality for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of L-infinity centrality over, or over all indices if rows=[]

        Returns
        -----------
        l_inf: ndarray. shape=(len(headers),)
            L-infinity centrality values computed considering all of the selected header variables

        NOTE: There should be no loops in this method!
        """
        data_selection = self.data.select_data(headers, rows)
        centralities = self.l_centrality(headers, rows, metric)
        maximum = np.argmax(centralities)
        return data_selection[np.argmax(centralities)], maximum


if __name__ == "__main__":
    # Set the data for analysis

    """
    filename = 'data/vertices.csv'
    vert_data = Data(filename)
    analysis = Analysis(vert_data)
    
    rowSet = np.random.randint(0, 1000000, 10000)
    centrality = analysis.l_centrality(["pos_x", "pos_y", "pos_z"], rows=rowSet, metric=Analysis.l2_norm)
    print(centrality)

    centrality = analysis.l_centrality(["pos_x", "pos_y", "pos_z"], rows=rowSet, metric=Analysis.l1_norm)
    print(centrality)

    centrality = analysis.l_centrality(["pos_x", "pos_y", "pos_z"], rows=rowSet, metric=Analysis.l_inf_norm)
    print(centrality)
    """
    val = Analysis.lp_norm(np.array([1, 2, 3]), np.array([2, 3, 4]), 3)
    print(f"Val: {val}")

    # centrality_alt = analysis.l_centrality_alt(["pos_x", "pos_y", "pos_z"], rows=rowSet, metric=Analysis.l2_norm)
    # print(centrality_alt)
