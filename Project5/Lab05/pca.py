"""
pca.py
Performs principal component analysis using the covariance matrix of the dataset
Roman Schiffino
CS 251 / 252: Data Analysis and Visualization
Spring 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data_transformations import normalize, center
from typing import List


class PCA:
    """Perform and store principal component analysis results

    NOTE: In your implementations, only the following "high level" `scipy`/`numpy` functions can be used:
    - `np.linalg.eig`
    The numpy functions that you have been using so far are fine to use.
    """

    def __init__(self, data_: pd.DataFrame):
        """

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset. Should be set as an instance variable.
        """
        self.data = data_

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

        # orig_means: ndarray. shape=(num_selected_vars,)
        #   Means of each orignal data variable
        self.orig_means = None

        # orig_mins: ndarray. shape=(num_selected_vars,)
        #   Mins of each orignal data variable
        self.orig_mins = None

        # orig_maxs: ndarray. shape=(num_selected_vars,)
        #   Maxs of each orignal data variable
        self.orig_maxs = None

    def get_prop_var(self):
        """(No changes should be needed)"""
        return self.prop_var

    def get_cum_var(self):
        """(No changes should be needed)"""
        return self.cum_var

    def get_eigenvalues(self):
        """(No changes should be needed)"""
        return self.e_vals

    def get_eigenvectors(self):
        """(No changes should be needed)"""
        return self.e_vecs

    @staticmethod
    def covariance_matrix(data_: np.ndarray):
        """Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`

        NOTE: You should do this wihout any loops
        NOTE: np.cov is off-limits here — compute it from "scratch"!
        """
        centered = center(data_)
        return np.einsum('ij,ik->jk', centered, centered) / (centered.shape[0] - 1)

    @staticmethod
    def compute_prop_var(e_vals: np.ndarray):
        """Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        """
        return (e_vals / e_vals.sum()).tolist()

    @staticmethod
    def compute_cum_var(prop_var: List[float]):
        """Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        """
        return np.cumsum(np.array(prop_var, dtype=np.float64)).tolist()

    @staticmethod
    def fit_static(data_: pd.DataFrame, vars_: List[str], normalize_dataset: bool = False):
        """Fits PCA to the data variables `vars` by computing the full set of PCs. The goal is to compute
        - eigenvectors and eigenvalues
        - proportion variance accounted for by each PC.
        - cumulative variance accounted for by first k PCs.

        Does NOT actually transform data by PCA.

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize_dataset: boolean.
            If True, min-max normalize each data variable it ranges from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.

        HINT:
        - It may be easier to convert to numpy ndarray format once selecting the appropriate data variables.
        - Before normalizing (if normalize_dataset is true), create instance variables containing information that would
        be needed to "undo" or reverse the normalization on the selected data.
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        - Remember, this method does NOT actually transform the dataset by PCA.
        """
        if isinstance(vars_, np.ndarray):
            pcs_to_keep = vars_.tolist()
        if data_ is None:
            raise ValueError('No data provided for PCA')
        if data_.__class__ != pd.DataFrame:
            raise ValueError('Data must be a pandas DataFrame')
        data_ = data_.copy()
        if vars_ is None:
            raise ValueError('No variables selected for PCA')
        if not isinstance(vars_, list):
            raise ValueError('Variables must be a list of strings')
        if len(vars_) < 1:
            raise ValueError('No variables selected for PCA')
        if len(vars_) == 1:
            raise ValueError('Too few variables selected for PCA')
        print(data_.columns.tolist())
        if any(var not in data_.columns.tolist() for var in vars_):
            raise ValueError('Variable names must match those used in the data DataFrame')
        if normalize_dataset is None:
            raise ValueError('No normalization parameter provided')
        ndSet = normalize_dataset
        A = data_.loc[:, vars_].to_numpy()
        orig_means = A.mean(axis=0)
        orig_mins = A.min(axis=0)
        orig_maxs = A.max(axis=0)
        if ndSet is True:
            A = normalize(A)
        else:
            pass
        means = A.mean(axis=0)
        mins = A.min(axis=0)
        maxs = A.max(axis=0)
        cov_mat = PCA.covariance_matrix(A)
        e_vals, e_vecs = np.linalg.eig(cov_mat)
        arg = np.argsort(e_vals)[::-1]
        e_vals = e_vals[arg]
        e_vecs = e_vecs[:, arg]
        prop_var = PCA.compute_prop_var(e_vals)
        cum_var = PCA.compute_cum_var(prop_var)
        return (vars_, A, orig_means, orig_mins, orig_maxs, means, mins, maxs, cov_mat, e_vals, e_vecs, prop_var,
                cum_var, ndSet)

    def fit(self, vars_: List[str], normalize_dataset: bool = False):
        """Fits PCA to the data variables `vars` by computing the full set of PCs. The goal is to compute
        - eigenvectors and eigenvalues
        - proportion variance accounted for by each PC.
        - cumulative variance accounted for by first k PCs.

        Does NOT actually transform data by PCA.

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize_dataset: boolean.
            If True, min-max normalize each data variable it ranges from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.

        HINT:
        - It may be easier to convert to numpy ndarray format once selecting the appropriate data variables.
        - Before normalizing (if normalize_dataset is true), create instance variables containing information that would
        be needed to "undo" or reverse the normalization on the selected data.
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        - Remember, this method does NOT actually transform the dataset by PCA.
        """
        (self.vars, self.A, self.orig_means, self.orig_mins, self.orig_maxs, means, mins, maxs, cov_mat, self.e_vals,
         self.e_vecs, self.prop_var, self.cum_var, self.normalized) = PCA.fit_static(self.data, vars_,
                                                                                     normalize_dataset)

    @staticmethod
    def elbow_plot_static(cumVar: List[float], num_pcs_to_keep: int = None):
        """
        Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        """
        if cumVar is None:
            raise ValueError('Cant plot cumulative variance. Compute the PCA first.')
        if num_pcs_to_keep is None:
            num_pcs_to_keep = len(cumVar)
        fig, ax = plt.subplots()
        ax.plot(np.arange(0, num_pcs_to_keep + 1), [0] + cumVar[:num_pcs_to_keep], marker='o', markersize=10)
        ax.set_xlabel('Number of Principal Components')
        ax.set_ylabel('Proportion Variance Accounted For')
        ax.set_title('Cumulative Variance Accounted For by Principal Components')
        ax.set_xticks(np.arange(0, num_pcs_to_keep + 1, 1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True)
        return fig, ax

    def elbow_plot(self, num_pcs_to_keep: int = None):
        """Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        """
        if self.cum_var is None:
            raise ValueError('Cant plot cumulative variance. Compute the PCA first.')
        return PCA.elbow_plot_static(self.cum_var, num_pcs_to_keep)

    @staticmethod
    def pca_project_static(A_: np.ndarray, eVecs: np.ndarray, pcs_to_keep: List[int]):
        """Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        eVecs: ndarray. shape=(num_vars, num_pcs)
            Eigenvectors of the covariance matrix of the data.
        data: ndarray. shape=(num_samps, num_vars)
            Data to project onto the PCs.
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        """
        if isinstance(pcs_to_keep, np.ndarray):
            pcs_to_keep = pcs_to_keep.tolist()
        if A_ is None:
            raise ValueError('No data provided for PCA')
        if eVecs is None:
            raise ValueError('No eigenvectors provided for PCA')
        if pcs_to_keep is None:
            raise ValueError('No PCs provided for PCA')
        if not isinstance(A_, np.ndarray):
            raise ValueError('Data must be a numpy ndarray')
        if not isinstance(eVecs, np.ndarray):
            raise ValueError('Eigenvectors must be a numpy ndarray')
        if A_.dtype != np.float64:
            raise ValueError('Data must be of type float64')
        if eVecs.dtype != np.float64:
            raise ValueError('Eigenvectors must be of type float64')
        if not isinstance(pcs_to_keep, list):
            raise ValueError('PC list must be a list')
        if any(not isinstance(val, int) for val in pcs_to_keep):
            raise ValueError('PC list must contain integers')
        return center(A_) @ eVecs[:, pcs_to_keep]

    def pca_project(self, pcs_to_keep: List[int]):
        """Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        """
        self.A_proj = PCA.pca_project_static(self.A, self.e_vecs, pcs_to_keep)
        return self.A_proj

    @staticmethod
    def pca_then_project_back_static(A_: np.ndarray, eVecs: np.ndarray, pcList: List[int], normed: bool = False,
                                     oMeans: np.ndarray = None, oMins: np.ndarray = None, oMaxs: np.ndarray = None):
        """
        Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars). Data to project onto the PCs.
        eVecs: ndarray. shape=(num_vars, num_pcs). Eigenvectors of the covariance matrix of the data.
        pcList: Python list of ints. len(pcList) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars). Data projected onto top K PCs then projected back to data space.

        NOTE: If you normalized, remember to rescale the data projected back to the original data space.
        """
        if not isinstance(normed, bool):
            raise ValueError('Normalization parameter must be a boolean')
        kwargs = [oMeans, oMins, oMaxs]
        if normed and any(val is None for val in kwargs):
            oMeans = A_.mean(axis=0)
            oMins = A_.min(axis=0)
            oMaxs = A_.max(axis=0)
        AProj = PCA.pca_project_static(A_, eVecs, pcList)
        injAProj = AProj @ eVecs[:, pcList].T + oMeans
        if normed:
            injAProj = injAProj * (oMaxs - oMins) + oMins
        return injAProj

    def pca_then_project_back(self, top_k: int):
        """Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        (Week 2)

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars). Data projected onto top K PCs then projected back to data space.

        NOTE: If you normalized, remember to rescale the data projected back to the original data space.
        """
        return PCA.pca_then_project_back_static(self.A, self.e_vecs, list(range(top_k)), self.normalized,
                                                self.orig_means,
                                                self.orig_mins, self.orig_maxs)

    def pca_then_project_back_alt(self, pcList: List[int]):
        """Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        (Week 2)

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars). Data projected onto top K PCs then projected back to data space.

        NOTE: If you normalized, remember to rescale the data projected back to the original data space.
        """
        return PCA.pca_then_project_back_static(self.A, self.e_vecs, pcList, self.normalized, self.orig_means,
                                                self.orig_mins, self.orig_maxs)

    @staticmethod
    def loading_plot_static(eVecs: np.ndarray, indexes: List[int], var_names: List[str] = None):
        """Create a loading plot of the slected eigenVectors in the array of PC eigenvectors

        (Week 2)
        """
        if isinstance(indexes, np.ndarray):
            indexes = indexes.tolist()
        if isinstance(var_names, np.ndarray):
            var_names = var_names.tolist()
        if eVecs is None:
            raise ValueError('No eigenvectors provided for PCA')
        if not isinstance(eVecs, np.ndarray):
            raise ValueError('Eigenvectors must be a numpy ndarray')
        if eVecs.dtype != np.float64:
            raise ValueError('Eigenvectors must be of type float64')
        if indexes is None:
            indexes = list(range(eVecs.shape[1]))
        if not isinstance(indexes, list):
            raise ValueError('Indexes must be a list')
        if any(not isinstance(val, int) for val in indexes):
            raise ValueError('Indexes must contain integers')
        if len(indexes) != 2:
            raise ValueError('Loading plot only supports 2D plots')
        if var_names is not None:
            if not isinstance(var_names, list):
                raise ValueError('Variable names must be a list')
            if any(not isinstance(val, str) for val in var_names):
                raise ValueError('Variable names must be strings')
            if len(var_names) != eVecs.shape[0]:
                raise ValueError('Variable names must match the number of variables in the data')
        else:
            var_names = [f'Var {i + 1}' for i in range(eVecs.shape[0])]
        indexes = indexes[:2]
        indexes.sort()
        selectedEVecs = eVecs[:, indexes]
        fig, ax = plt.subplots()
        max_x = 0
        max_y = 0
        for i in range(selectedEVecs.shape[0]):
            cur_x = selectedEVecs[i, 0]
            abs_x = abs(cur_x)
            if abs_x > max_x:
                max_x = abs_x
            cur_y = selectedEVecs[i, 1]
            abs_y = abs(cur_y)
            if abs_y > max_y:
                max_y = abs_y

            ax.plot([0, cur_x], [0, cur_y], label=f"{var_names[i]}", marker='o',
                    markersize=5)
            ax.annotate(f'{var_names[i]}', (cur_x, cur_y))
        max_x = 1.1 * max_x
        max_y = 1.1 * max_y
        ax.set_xlabel(f'PC {indexes[0] + 1}')
        ax.set_ylabel(f'PC {indexes[1] + 1}')
        ax.set_xlim(-max_x, max_x)
        ax.set_ylim(-max_y, max_y)
        length = len(indexes)
        if indexes == list(range(length)):
            ax.set_title(f'Loading Plot of Top {length} PC Eigenvectors')
        else:
            numbers = [str(i + 1) for i in indexes]
            ax.set_title(f'Loading Plot of {numbers[0]} and {numbers[1]} PC Eigenvectors')
        ax.grid(True)
        ax.legend()
        return fig, ax

    def loading_plot(self):
        """Create a loading plot of the top 2 PC eigenvectors

        (Week 2)

        TODO:
        - Plot a line joining the origin (0, 0) and corresponding components of the top 2 PC eigenvectors.
            Example: If e_0 = [0.1, 0.3] and e_1 = [1.0, 2.0], you would create two lines to join
            (0, 0) and (0.1, 1.0); (0, 0) and (0.3, 2.0).
            Number of lines = num_vars
        - Use plt.annotate to label each line by the variable that it corresponds to.
        - Reminder to create useful x and y axis labels.
        """
        return PCA.loading_plot_static(self.e_vecs, [0, 1], self.vars)


if __name__ == '__main__':
    # Load the data
    data = pd.read_csv('data/iris.csv')

    # Create an instance of PCA
    pca = PCA(data)

    # Test fitting the PCA
    pca.fit(vars_=['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'], normalize_dataset=True)

    # Test get_prop_var
    prop_var = pca.get_prop_var()
    print(f"prop_var: {prop_var}")
    assert isinstance(prop_var, list), "get_prop_var should return a list"
    print("get_prop_var passed")

    # Test get_cum_var
    cum_var = pca.get_cum_var()
    print(f"cum_var: {cum_var}")
    assert isinstance(cum_var, list), "get_cum_var should return a list"
    print("get_cum_var passed")

    # Test get_eigenvalues
    e_vals = pca.get_eigenvalues()
    print(f"e_vals: {e_vals}")
    assert isinstance(e_vals, np.ndarray), "get_eigenvalues should return a numpy ndarray"
    print("get_eigenvalues passed")

    # Test get_eigenvectors
    e_vecs = pca.get_eigenvectors()
    print(f"e_vecs: {e_vecs}")
    assert isinstance(e_vecs, np.ndarray), "get_eigenvectors should return a numpy ndarray"
    print("get_eigenvectors passed")

    # Test covariance_matrix
    cov_matrix = pca.covariance_matrix(data.values)
    print(f"cov_matrix: {cov_matrix}")
    assert isinstance(cov_matrix, np.ndarray), "covariance_matrix should return a numpy ndarray"
    print("covariance_matrix passed")

    # Test projection
    proj = pca.pca_project([0, 1])
    print(f"projection: {proj}")
    assert isinstance(proj, np.ndarray), "projection should return a numpy ndarray"
    print("projection passed")

    # Test pca_then_project_back
    proj_back = pca.pca_then_project_back(2)
    print(f"projection_back: {proj_back}")
    assert isinstance(proj_back, np.ndarray), "pca_then_project_back should return a numpy ndarray"
    print("pca_then_project_back passed")

    # Test pca_then_project_back_alt
    proj_back_alt = pca.pca_then_project_back_alt([0, 1, 2])
    print(f"projection_back_alt: {proj_back_alt}")
    assert isinstance(proj_back_alt, np.ndarray), "pca_then_project_back_alt should return a numpy ndarray"
    print("pca_then_project_back_alt passed")


    # Test variance_accounted_for
    prop_var = pca.compute_prop_var(e_vals)
    print(f"prop_var: {prop_var}")
    assert isinstance(prop_var, list), "compute_prop_var should return a list"
    print("compute_prop_var passed")

    # Test cumulative_variance_accounted_for
    cum_var = pca.compute_cum_var(prop_var)
    print(f"cum_var: {cum_var}")
    assert isinstance(cum_var, list), "compute_cum_var should return a list"
    print("compute_cum_var passed")

    # Test elbow_plot
    fig, ax = pca.elbow_plot()
    assert isinstance(fig, plt.Figure), "elbow_plot should return a matplotlib Figure"
    assert isinstance(ax, plt.Axes), "elbow_plot should return a matplotlib Axes"
    plt.show()
    print("elbow_plot passed")

    # Test loading_plot
    fig2, ax = pca.loading_plot()
    assert isinstance(fig2, plt.Figure), "loading_plot should return a matplotlib Figure"
    assert isinstance(ax, plt.Axes), "loading_plot should return a matplotlib Axes"
    plt.show()
    print("loading_plot passed")

    # Test loading_plot_static
    fig3, ax = PCA.loading_plot_static(e_vecs, [0, 2], ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth'])
    assert isinstance(fig3, plt.Figure), "loading_plot_static should return a matplotlib Figure"
    assert isinstance(ax, plt.Axes), "loading_plot_static should return a matplotlib Axes"
    plt.show()
    print("loading_plot_static passed")

    print("All tests passed!")
