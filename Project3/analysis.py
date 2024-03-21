"""analysis.py
Run statistical analyses and plot Numpy ndarray data
YOUR NAME HERE
CS 251/2: Data Analysis and Visualization
Spring 2024
"""
import matplotlib.pyplot as plt

from data import *


def show():
    """Simple wrapper function for matplotlib's show function.

    (Does not require modification)
    """
    plt.show()


class Analysis:
    def __init__(self, data):
        """

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        """
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    @staticmethod
    def l1_norm(a: np.ndarray, b: np.ndarray):
        if a.shape != b.shape and a.shape != b.T.shape:
            raise ValueError("Arrays must have the same shape")
        if a.ndim != 1:
            raise ValueError("Arrays must be 1D")
        if np.array_equal(a, b):
            return 0
        return np.sum(np.abs(b - a), axis=0)

    @staticmethod
    def l2_norm(a: np.ndarray, b: np.ndarray):
        if a.shape != b.shape and a.shape != b.T.shape:
            raise ValueError("Arrays must have the same shape")
        if a.ndim != 1:
            raise ValueError("Arrays must be 1D")
        if np.array_equal(a, b):
            return 0
        c = b - a
        return np.sqrt(np.einsum('i,i->', c, c))

    @staticmethod
    def lp_norm(a: np.ndarray, b: np.ndarray, p: int):
        if a.shape != b.shape and a.shape != b.T.shape:
            raise ValueError("Arrays must have the same shape")
        if a.ndim != 1:
            raise ValueError("Arrays must be 1D")
        if np.array_equal(a, b):
            return 0
        pfloat = float(p)
        inv_p = float(float(1) / pfloat)
        return np.float_power(np.sum(np.abs(b - a) ** pfloat, axis=0), inv_p)

    @staticmethod
    def l_inf_norm(a: np.ndarray, b: np.ndarray):
        if a.shape != b.shape and a.shape != b.T.shape:
            raise ValueError("Arrays must have the same shape")
        if a.ndim != 1:
            raise ValueError("Arrays must be 1D")
        if np.array_equal(a, b):
            return 0
        return np.max(np.abs(b - a), axis=0)

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

        TODO:
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

    # centrality_alt = analysis.l_centrality_alt(["pos_x", "pos_y", "pos_z"], rows=rowSet, metric=Analysis.l2_norm)
    # print(centrality_alt)
