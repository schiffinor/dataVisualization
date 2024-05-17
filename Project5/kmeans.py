"""kmeans.py
Performs K-Means clustering
YOUR NAME HERE
CS 251/2: Data Analysis and Visualization
Spring 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Union, Dict
from Lab05 import analysis


class KMeans:
    def __init__(self, data: np.ndarray = None):
        """
        KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        """

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None

        if data is not None:
            # data: ndarray. shape=(num_samps, num_features)
            if not isinstance(data, np.ndarray):
                raise TypeError('data must be a numpy array')
            if data.ndim != 2:
                raise ValueError('data must be a 2D numpy array')
            if data.shape[0] < 1 or data.shape[1] < 1:
                raise ValueError('data must have at least one sample and one feature')
            self.data = data.copy()
            self.num_samps, self.num_features = data.shape

    def set_data(self, data: np.ndarray):
        """Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        """
        if not isinstance(data, np.ndarray):
            raise TypeError('data must be a numpy array')
        if data.ndim != 2:
            raise ValueError('data must be a 2D numpy array')
        if data.shape[0] < 1 or data.shape[1] < 1:
            raise ValueError('data must have at least one sample and one feature')
        self.data = data.copy()
        self.num_samps, self.num_features = data.shape

    def get_data(self):
        """Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        """
        return self.data.copy()

    def get_centroids(self):
        """Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        """
        return self.centroids

    def get_data_centroid_labels(self):
        """Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        """
        return self.data_centroid_labels

    @staticmethod
    def dist_pt_to_pt(pt_1: np.ndarray, pt_2: np.ndarray = None, norm: staticmethod = analysis.Analysis.l2_norm,
                      normP: int = 2) -> float:
        """Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        """
        if pt_1 is None:
            raise ValueError('pt_1 cannot be None')
        if pt_1.ndim != 1:
            raise ValueError('pt_1 must be a 1D numpy array')
        if pt_2 is None:
            pt_2 = np.zeros_like(pt_1)
        if norm is None:
            raise ValueError('norm cannot be None')
        if not callable(norm):
            raise ValueError('norm must be a callable function')
        if norm.__name__ not in ['l1_norm', 'l2_norm', 'l_inf_norm']:
            if norm.__name__ == 'l_p_norm':
                if not isinstance(normP, int):
                    raise ValueError('normP must be an integer')
                if normP < 2:
                    raise ValueError('normP must be greater than or equal to 2')
                return norm(pt_1, pt_2, normP)
            else:
                raise ValueError('norm must be one of l1_norm, l2_norm, l_inf_norm, or l_p_norm')
        return norm(pt_1, pt_2)


    @staticmethod
    def ndim_norm_choice_and_call(arr1: np.ndarray, arr2: np.ndarray,
                                    norm: staticmethod = analysis.Analysis.l2_norm,
                                    normP: int = 2) -> np.ndarray:
        """
        Choose a norm function and call it with the appropriate arguments.
        Utility function to help with cut down on duped code.
        :param arr1: np.ndarray. shape=(N, M)
        :param arr2: np.ndarray. shape=(N, M)
        :param norm: staticmethod. A norm function to call.
        :param normP: int. The p value for the norm function.

        :return: Distance between arr1 and arr2 using the norm function.
        """
        if arr1 is None:
            raise ValueError('arr1 cannot be None')
        if arr1.ndim != 2:
            raise ValueError('arr1 must be a 2D numpy array')
        if arr2 is None:
            raise ValueError('arr2 cannot be None')
        if arr2.ndim != 2:
            raise ValueError('arr2 must be a 2D numpy array')
        if arr1.shape != arr2.shape:
            raise ValueError('arr1 and arr2 must have the same shape')
        if norm is None:
            raise ValueError('norm cannot be None')
        if not callable(norm):
            raise ValueError('norm must be a callable function')

        norm_names = [analysis.Analysis.l1_norm.__name__,
                      analysis.Analysis.l2_norm.__name__,
                      analysis.Analysis.l_inf_norm.__name__]
        ndim_norms = [analysis.Analysis.l1_norm_ndim,
                      analysis.Analysis.l2_norm_ndim,
                      analysis.Analysis.l_inf_norm_ndim]
        ndim_norm_dic = dict(zip(norm_names, ndim_norms))
        if norm.__name__ in norm_names:
            ndim_norms = ndim_norm_dic[norm.__name__]
            return ndim_norms(arr1, arr2)
        else:
            if norm.__name__ == 'l_p_norm':
                if not isinstance(normP, int):
                    raise ValueError('normP must be an integer')
                if normP < 2:
                    raise ValueError('normP must be greater than or equal to 2')
                return analysis.Analysis.lp_norm_ndim(arr1, arr2, normP)
            else:
                raise ValueError('norm must be one of l1_norm, l2_norm, l_inf_norm, or l_p_norm')


    @staticmethod
    def dist_pt_to_centroids_static(pt: np.ndarray, centroids: np.ndarray,
                                    norm: staticmethod = analysis.Analysis.l2_norm,
                                    normP: int = 2) -> np.ndarray:
        """
        Compute the Euclidean distance between data sample `pt` and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        """
        if pt is None:
            raise ValueError('pt cannot be None')
        if pt.ndim != 1:
            raise ValueError('pt must be a 1D numpy array')
        if centroids is None:
            raise ValueError('centroids cannot be None')
        if centroids.ndim != 2:
            raise ValueError('centroids must be a 2D numpy array')
        if centroids.shape[1] != pt.shape[0]:
            raise ValueError('centroids and pt must have the same number of features')
        if norm is None:
            raise ValueError('norm cannot be None')
        if not callable(norm):
            raise ValueError('norm must be a callable function')

        pt_arr = np.vstack([pt for i in range(centroids.shape[0])])
        return KMeans.ndim_norm_choice_and_call(pt_arr, centroids, norm, normP)




    def dist_pt_to_centroids(self, pt: np.ndarray, centroids: np.ndarray = None,
                             norm: staticmethod = analysis.Analysis.l2_norm,
                             normP: int = 2) -> np.ndarray:
        """
        Compute the Euclidean distance between data sample `pt` and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        """
        if centroids is None:
            centroids = self.centroids
        return self.dist_pt_to_centroids_static(pt, centroids, norm, normP)


    @staticmethod
    def dist_pts_to_centroids_static(pts: np.ndarray, centroids: np.ndarray = None,
                                     norm: staticmethod = analysis.Analysis.l2_norm,
                                     normP: int = 2) -> np.ndarray:
        """
         Compute the Euclidean distance between data samples `pts` and all the cluster centroids
         self.centroids

         Parameters:
         -----------
         pt: ndarray. shape=(num_features,)
         centroids: ndarray. shape=(C, num_features)
             C centroids, where C is an int.

         Returns:
         -----------
         ndarray. shape=(C,).
             distance between pt and each of the C centroids in `centroids`.

         NOTE: Implement without any for loops (you will thank yourself later since you will wait
         only a small fraction of the time for your code to stop running)
         """
        if pts is None:
            raise ValueError('pts cannot be None')
        if pts.ndim != 2:
            raise ValueError('pts must be a 2D numpy array')
        if centroids is None:
            raise ValueError('centroids cannot be None')
        if centroids.ndim != 2:
            raise ValueError('centroids must be a 2D numpy array')
        if centroids.shape[1] != pts.shape[1]:
            raise ValueError('centroids and pts must have the same number of features')
        if norm is None:
            raise ValueError('norm cannot be None')
        if not callable(norm):
            raise ValueError('norm must be a callable function')

        pts_arr = np.vstack([pt for i in range(centroids.shape[0]) for pt in pts])
        centroids_arr = np.vstack([centroids for i in range(pts.shape[0])])

        return KMeans.ndim_norm_choice_and_call(pts_arr, centroids_arr, norm, normP)


    def dist_pts_to_centroids(self, pts: np.ndarray = None, centroids: np.ndarray = None,
                                norm: staticmethod = analysis.Analysis.l2_norm,
                                normP: int = 2) -> np.ndarray:
            """
            Compute the Euclidean distance between data samples `pts` and all the cluster centroids
            self.centroids

            Parameters:
            -----------
            pts: ndarray. shape=(num_samps, num_features)
            centroids: ndarray. shape=(C, num_features)
                C centroids, where C is an int.

            Returns:
            -----------
            ndarray. shape=(C,).
                distance between pt and each of the C centroids in `centroids`.

            NOTE: Implement without any for loops (you will thank yourself later since you will wait
            only a small fraction of the time for your code to stop running)
            """
            if pts is None:
                pts = self.data
            if centroids is None:
                centroids = self.centroids
            return self.dist_pts_to_centroids_static(pts, centroids, norm, normP)


    @staticmethod
    def initialize_static(data_: np.ndarray, k: int = 1) -> Tuple[np.ndarray, int]:
        """
        Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        """
        if data_ is None:
            raise ValueError('data_ cannot be None')
        if data_.ndim != 2:
            raise ValueError('data_ must be a 2D numpy array')
        if not isinstance(k, int):
            raise ValueError('k must be an integer')
        if k < 1:
            raise ValueError('k must be greater than or equal to 1')
        if k > data_.shape[0]:
            raise ValueError('k must be less than or equal to the number of data samples')
        return k, data_[np.random.choice(data_.shape[0], k, replace=False)]


    def initialize(self, k: int = 1) -> np.ndarray:
        """
        Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        """
        output, self.k = KMeans.initialize_static(self.data, k)
        return output


    @staticmethod
    def cluster_static(data_: np.ndarray, k: int = 2, tol: float = 1e-2, max_iter: int = 1000,
                       verbose: bool = False, p: int = 2, norm: staticmethod = analysis.Analysis.l2_norm,
                       normP: int = 2) -> Tuple[int, float, np.ndarray, np.ndarray]:
        """
        Performs K-means clustering on the data

        :param data_: ndarray. shape=(num_samps, num_features).
            The dataset to be clustered
        :param k: int.
            Number of clusters
        :param tol: float.
            Terminate K-means if the (absolute value of) the difference between all the centroid values from the previous and current time step < `tol`.
        :param max_iter: int.
            Make sure that K-means does not run more than `max_iter` iterations.
        :param verbose: boolean.
            Print out debug information if set to True.
        :param p: int.
            The p value for the norm function.


        :return int. Number of iterations that K-means was run for
        :return inertia. float. Mean squared distance between each data sample and its cluster mean
        :return
        :return

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        """
        if self.num_samps < k:
            raise RuntimeError('Cannot compute kmeans with #data samples < k!')
        if k < 1:
            raise RuntimeError('Cannot compute kmeans with k < 1!')

    def cluster_batch(self, k=2, n_iter=1, verbose=False, p=2):
        """Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        """
        pass

    def update_labels(self, centroids):
        """Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        """
        pass

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        """Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        """
        pass

    def compute_inertia(self):
        """Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        """
        pass

    def plot_clusters(self):
        """Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). To make sure you change your colors to be clearly differentiable,
        use either the Okabe & Ito or one of the Petroff color palettes: https://github.com/proplot-dev/proplot/issues/424
        Each string in the `colors` list that starts with # is the hexadecimal representation of a color (blue, red, etc.)
        that can be passed into the color `c` keyword argument of plt.plot or plt.scatter.
            Pick one of the palettes with a generous number of colors so that you don't run out if k is large (e.g. >6).
        """
        pass

    def elbow_plot(self, max_k):
        """Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        """
        pass

    def replace_color_with_centroid(self):
        """Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        """
        pass
