"""kmeans.py
Performs K-Means clustering
YOUR NAME HERE
CS 251/2: Data Analysis and Visualization
Spring 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Union, Dict, NewType, Callable
import analysis

# Define a type for a function that takes two np.ndarrays and returns a scalar
Norm = NewType("Norm", Callable[[np.ndarray, np.ndarray, ...], float])


class KMeans:
    def __init__(self, data: np.ndarray = None):
        """
        KMeans constructor

        (Should not require any changes)

        :param data: ndarray. shape=(num_samps, num_features)
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

        :param data: ndarray. shape=(num_samps, num_features)
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

        :return ndarray. shape=(num_samps, num_features). COPY of the data
        """
        return self.data.copy()

    def get_centroids(self):
        """Get the K-means centroids

        (Should not require any changes)

        :return ndarray. shape=(k, self.num_features).
        """
        return self.centroids

    def get_data_centroid_labels(self):
        """Get the data-to-cluster assignments

        (Should not require any changes)

        :return ndarray of ints. shape=(self.num_samps,)
        """
        return self.data_centroid_labels

    @staticmethod
    def norm_validation(norm: Norm, normP: float = 2):
        if norm is None:
            raise ValueError('norm cannot be None')
        if not callable(norm):
            raise ValueError('norm must be a callable function')
        if norm.__name__ not in ['l1_norm', 'l2_norm', 'l_inf_norm', 'l0_norm']:
            if norm.__name__ == 'lp_norm':
                if not isinstance(normP, int | float):
                    raise ValueError('normP must be an int or float')
                if normP < 0:
                    raise ValueError('normP must be greater than or equal to 0')
                pass
            else:
                raise ValueError('norm must be one of l1_norm, l2_norm, l_inf_norm, or lp_norm')
        pass

    @staticmethod
    def dist_pt_to_pt(pt_1: np.ndarray, pt_2: np.ndarray = None, norm: Norm = analysis.Analysis.l2_norm,
                      normP: float = 2) -> float:
        """
        Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        :param pt_1: ndarray. shape=(num_features,)
            The first data sample
        :param pt_2: ndarray. shape=(num_features,)
            The second data sample
        :param norm: staticmethod.
            A norm function to call.
        :param normP: float.
            The p value for the norm function.

        :return distance: float.
            Normed distance between pt_1 and pt_2

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
        KMeans.norm_validation(norm, normP)
        return norm(pt_1, pt_2)

    @staticmethod
    def ndim_norm_choice_and_call(arr1: np.ndarray, arr2: np.ndarray,
                                  norm: Norm = analysis.Analysis.l2_norm,
                                  normP: float = 2) -> np.ndarray:
        """
        Choose a norm function and call it with the appropriate arguments.
        Utility function to help with cut down on duped code.
        :param arr1: np.ndarray. shape=(N, M)
        :param arr2: np.ndarray. shape=(N, M)
        :param norm: staticmethod. A norm function to call.
        :param normP: float. The p value for the norm function.

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
        KMeans.norm_validation(norm, normP)

        norm_names = [analysis.Analysis.l1_norm.__name__,
                      analysis.Analysis.l2_norm.__name__,
                      analysis.Analysis.l_inf_norm.__name__,
                      analysis.Analysis.l0_norm.__name__]
        ndim_norms = [analysis.Analysis.l1_norm_ndim,
                      analysis.Analysis.l2_norm_ndim,
                      analysis.Analysis.l_inf_norm_ndim,
                      analysis.Analysis.l0_norm_ndim]
        ndim_norm_dic = dict(zip(norm_names, ndim_norms))
        if norm.__name__ in norm_names:
            ndim_norms = ndim_norm_dic[norm.__name__]
            return ndim_norms(arr1, arr2)
        else:
            if norm.__name__ == 'lp_norm':
                if not isinstance(normP, int | float):
                    raise ValueError('normP must be an integer or float')
                if normP < 0:
                    raise ValueError('normP must be greater than or equal to 0')
                return analysis.Analysis.lp_norm_ndim(arr1, arr2, normP)
            else:
                raise ValueError('norm must be one of l1_norm, l2_norm, l_inf_norm, l0_norm, or lp_norm')

    @staticmethod
    def dist_pt_to_pts(pt: np.ndarray, pts: np.ndarray,
                       norm: Norm = analysis.Analysis.l2_norm,
                       normP: float = 2) -> np.ndarray:
        """
        Compute the Euclidean distance between data sample `pt` and all the pts centroids
        self.centroids

        :param pt: ndarray. shape=(num_features,)
            The point to compute the distance from the centroids.
        :param pts: ndarray. shape=(C, num_features)
            C pts, where C is an int.
        :param norm: staticmethod.
            A norm function to call.
        :param normP: float.
            The p value for the norm function.

        :returns distances: ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        """
        if pt is None:
            raise ValueError('pt cannot be None')
        if pt.ndim != 1:
            raise ValueError('pt must be a 1D numpy array')
        if pts is None:
            raise ValueError('centroids cannot be None')
        if pts.ndim != 2:
            raise ValueError('centroids must be a 2D numpy array')
        if pts.shape[1] != pt.shape[0]:
            raise ValueError('centroids and pt must have the same number of features')
        KMeans.norm_validation(norm, normP)

        pt_arr = np.vstack([pt for _ in range(pts.shape[0])])
        return KMeans.ndim_norm_choice_and_call(pt_arr, pts, norm, normP)

    def dist_pt_to_centroids(self, pt: np.ndarray, centroids: np.ndarray = None,
                             norm: Norm = analysis.Analysis.l2_norm,
                             normP: float = 2) -> np.ndarray:
        """
        Compute the Euclidean distance between data sample `pt` and all the cluster centroids
        self.centroids

        :param pt: ndarray. shape=(num_features,)
            The point to compute the distance from the centroids.
        :param centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.
        :param norm: staticmethod.
            A norm function to call.
        :param normP: float.
            The p value for the norm function.

        :return distances: ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        """
        if centroids is None:
            centroids = self.centroids
        return self.dist_pt_to_pts(pt, centroids, norm, normP)

    @staticmethod
    def dist_pts_to_pts(pts: np.ndarray, pts2: np.ndarray = None,
                        norm: Norm = analysis.Analysis.l2_norm,
                        normP: float = 2) -> np.ndarray:
        """
         Compute the Euclidean distance between data samples `pts` and all pts in `pts2`

         :param pts: ndarray. shape=(numsamps, num_features,)
            The points to compute the distance from the centroids.
         :param pts2: ndarray. shape=(C, num_features)
            C centroids, where C is an int.
        :param norm: staticmethod.
            A norm function to call.
        :param normP: float.
            The p value for the norm function.

         :return distances: ndarray. shape=(numsamps, C,).
             distance between pt and each of the C centroids in `centroids`.

         NOTE: Implement without any for loops (you will thank yourself later since you will wait
         only a small fraction of the time for your code to stop running)
         """
        if pts is None:
            raise ValueError('pts cannot be None')
        if pts.ndim != 2:
            raise ValueError('pts must be a 2D numpy array')
        if pts2 is None:
            raise ValueError('pts2 cannot be None')
        if pts2.ndim != 2:
            raise ValueError('pts2 must be a 2D numpy array')
        if pts2.shape[1] != pts.shape[1]:
            raise ValueError('pts2 and pts must have the same number of features')
        KMeans.norm_validation(norm, normP)

        pts_arr_list = [pts[i] for i in range(pts.shape[0]) for _ in range(pts2.shape[0])]
        pts_arr = np.vstack(pts_arr_list)
        pts2_arr = np.vstack([pts2 for _ in range(pts.shape[0])])
        return KMeans.ndim_norm_choice_and_call(pts_arr, pts2_arr, norm, normP).reshape(pts.shape[0], pts2.shape[0])

    def dist_pts_to_centroids(self, pts: np.ndarray = None, centroids: np.ndarray = None,
                              norm: Norm = analysis.Analysis.l2_norm,
                              normP: float = 2) -> np.ndarray:
        """
        Compute the Euclidean distance between data samples `pts` and all the cluster centroids
        self.centroids

        :param pts: ndarray. shape=(num_samps, num_features)
            The points to compute the distance from the centroids.
        :param centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.
        :param norm: staticmethod.
            A norm function to call.
        :param normP: float.
            The p value for the norm function.

        :return distances: ndarray. shape=(num_samps, C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        """
        if pts is None:
            pts = self.data
        if centroids is None:
            centroids = self.centroids
        return self.dist_pts_to_pts(pts, centroids, norm, normP)

    @staticmethod
    def initialize_static(data_: np.ndarray, k: int = 1) -> Tuple[np.ndarray, int]:
        """
        Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        :param data_: ndarray. shape=(num_samps, num_features).
            The dataset to be clustered
        :param k: int.
            Number of clusters

        :return data_selection: ndarray. shape=(k, self.num_features).
            Initial centroids for the k clusters.
        :return k: int.
            Number of clusters

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
        return data_[np.random.choice(data_.shape[0], k, replace=False)], k

    def initialize(self, k: int = 1) -> np.ndarray:
        """
        Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples


        :param k: int.
            Number of clusters

        :return output: ndarray. shape=(k, self.num_features).
            Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        """
        output, self.k = KMeans.initialize_static(self.data, k)
        return output

    @staticmethod
    def cluster_static(data_: np.ndarray, k: int = 2, tol: float = 1e-8, max_iter: int = 1000,
                       verbose: bool = False, norm: Norm = analysis.Analysis.l2_norm,
                       normP: float = 2) -> Tuple[int, float, np.ndarray, np.ndarray]:
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
        :param norm: staticmethod.
            A norm function to call.
        :param normP: float.
            The p value for the norm function.


        :return i. int.
            Number of iterations that K-means was run for
        :return inertia. float.
            Mean squared distance between each data sample and its cluster mean
        :return centroids. ndarray. shape=(k, num_features).
            Centroids for each cluster
        :return data_centroid_labels. ndarray of ints. shape=(num_samps,).
            Holds index of the assigned cluster of each data sample

        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        """
        if data_.shape[0] < k:
            raise RuntimeError('Cannot compute kmeans with #data samples < k!')
        if k < 1:
            raise RuntimeError('Cannot compute kmeans with k < 1!')
        if tol < 0:
            raise RuntimeError('Cannot compute kmeans with tol < 0!')
        if max_iter < 1:
            raise RuntimeError('Cannot compute kmeans with max_iter < 1!')
        if verbose is None:
            raise ValueError('verbose cannot be None')
        if not isinstance(verbose, bool):
            raise ValueError('verbose must be a boolean')
        KMeans.norm_validation(norm, normP)

        centroids, k = KMeans.initialize_static(data_, k)
        iter_count = 0
        data_centroid_labels = None
        for i in range(max_iter):
            data_centroid_labels = KMeans.update_labels_static(data_, centroids)
            centroids, centroid_diff = KMeans.update_centroids_static(data_, k, data_centroid_labels, centroids)
            if np.all(np.abs(centroid_diff) < tol):
                break
        inertia = KMeans.compute_inertia_static(data_, centroids, data_centroid_labels)
        return iter_count, inertia, centroids, data_centroid_labels

    def cluster(self, k: int = 2, tol: float = 1e-4, max_iter: int = 1000,
                verbose: bool = False, norm: Norm = analysis.Analysis.l2_norm,
                normP: float = 2) -> Tuple[int, float, np.ndarray, np.ndarray]:
        """
        Performs K-means clustering on the data

        :param k: int.
            Number of clusters
        :param tol: float.
            Terminate K-means if the (absolute value of) the difference between all the centroid values from the previous and current time step < `tol`.
        :param max_iter: int.
            Make sure that K-means does not run more than `max_iter` iterations.
        :param verbose: boolean.
            Print out debug information if set to True.
        :param norm: staticmethod.
            A norm function to call.
        :param normP: float.
            The p value for the norm function.


        :return i: int.
            Number of iterations that K-means was run for
        :return inertia: float.
            Mean squared distance between each data sample and its cluster mean
        :return centroids: ndarray. shape=(k, num_features).
            Centroids for each cluster
        :return data_centroid_labels. ndarray of ints. shape=(num_samps,).
            Holds index of the assigned cluster of each data sample
        """
        i, inertia, centroids, data_centroid_labels = KMeans.cluster_static(self.data, k, tol, max_iter, verbose, norm,
                                                                            normP)
        self.k = k
        self.centroids = centroids
        self.data_centroid_labels = data_centroid_labels
        self.inertia = inertia
        if verbose:
            print(f"KMeans.cluster: i: {i}, inertia: {inertia}.")
        return i, inertia, centroids, data_centroid_labels

    @staticmethod
    def cluster_batch_static(data_: np.ndarray, k: int = 2, n_iter: int = 1, verbose: bool = False,
                             norm: Norm = analysis.Analysis.l2_norm,
                             normP: float = 2) -> Tuple[int, float, np.ndarray, np.ndarray]:
        """
        Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia.

        :param data_: ndarray. shape=(num_samps, num_features).
            The dataset to be clustered
        :param k: int.
            Number of clusters
        :param n_iter: int.
            Number of times to run K-means with the designated `k` value.
        :param verbose: boolean.
            Print out debug information if set to True.
        :param norm: staticmethod.
            A norm function to call.
        :param normP: float.
            The p value for the norm function.

        :return i: int.
            Number of iterations that K-means was run for
        :return inertia: float.
            Mean squared distance between each data sample and its cluster mean
        :return centroids: ndarray. shape=(k, num_features).
            Centroids for each cluster
        :return data_centroid_labels. ndarray of ints. shape=(num_samps,).
            Holds index of the assigned cluster of each data sample
        """
        if n_iter < 1:
            raise ValueError('n_iter must be greater than or equal to 1')
        best_inertia = np.inf
        best_run = None
        for i in range(n_iter):
            i, inertia, centroids, data_centroid_labels = KMeans.cluster_static(data_, k, verbose=verbose, norm=norm,
                                                                                normP=normP)
            if inertia < best_inertia:
                best_inertia = inertia
                best_run = i, inertia, centroids, data_centroid_labels
        if verbose:
            print(f"KMeans.cluster: i: {best_run[0]}, inertia: {best_run[1]}.")
        return best_run

    def cluster_batch(self, k: int = 2, n_iter: int = 1, verbose: bool = False,
                      norm: Norm = analysis.Analysis.l2_norm,
                      normP: float = 2) -> Tuple[int, float, np.ndarray, np.ndarray]:
        """
        Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        :param k: int.
            Number of clusters
        :param n_iter: int.
            Number of times to run K-means with the designated `k` value.
        :param verbose: boolean.
            Print out debug information if set to True.
        :param norm: staticmethod.
            A norm function to call.
        :param normP: float.
            The p value for the norm function.

        :return i: int.
            Number of iterations that K-means was run for
        :return inertia: float.
            Mean squared distance between each data sample and its cluster mean
        :return centroids: ndarray. shape=(k, num_features).
            Centroids for each cluster
        :return data_centroid_labels. ndarray of ints. shape=(num_samps,).
            Holds index of the assigned cluster of each data sample
        """
        i, inertia, centroids, data_centroid_labels = KMeans.cluster_batch_static(self.data, k, n_iter, verbose, norm,
                                                                                  normP)
        self.k = k
        self.centroids = centroids
        self.data_centroid_labels = data_centroid_labels
        self.inertia = inertia
        return i, inertia, centroids, data_centroid_labels

    @staticmethod
    def update_labels_static(data_: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assigns each data sample to the nearest centroid

        :param data_: ndarray. shape=(num_samps, num_features).
            The dataset to be clustered
        :param centroids: ndarray. shape=(k, self.num_features).
            Current centroids for the k clusters.

        :return labels: ndarray of ints. shape=(self.num_samps,).
            Holds index of the assigned cluster of each data sample.
            These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        """
        if data_ is None:
            raise ValueError('data_ cannot be None')
        if data_.ndim != 2:
            raise ValueError('data_ must be a 2D numpy array')
        if centroids is None:
            raise ValueError('centroids cannot be None')
        if centroids.ndim != 2:
            raise ValueError('centroids must be a 2D numpy array')
        if data_.shape[1] != centroids.shape[1]:
            raise ValueError('data_ and centroids must have the same number of features')
        return np.argmin(KMeans.dist_pts_to_pts(data_, centroids), axis=1)

    def update_labels(self, centroids: np.ndarray = None) -> np.ndarray:
        """Assigns each data sample to the nearest centroid

        :param centroids: ndarray. shape=(k, self.num_features).
            Current centroids for the k clusters.

        :return labels: ndarray of ints. shape=(self.num_samps,).
            Holds index of the assigned cluster of each data sample.
            These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        """
        if centroids is None:
            centroids = self.centroids
        return KMeans.update_labels_static(self.data, centroids)

    @staticmethod
    def update_centroids_static(data_: np.ndarray, k: int, data_centroid_labels: np.ndarray = None,
                                prev_centroids: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes each of the K centroids (means) based on the data assigned to each cluster

        :param data_: ndarray. shape=(num_samps, num_features).
            The dataset to be clustered
        :param k: int.
            Number of clusters
        :param data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        :param prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        :return new_centroids: ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        :return centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        """
        if data_ is None:
            raise ValueError('data_ cannot be None')
        if data_.ndim != 2:
            raise ValueError('data_ must be a 2D numpy array')
        if not isinstance(k, int):
            raise ValueError('k must be an integer')
        if k < 1:
            raise ValueError('k must be greater than or equal to 1')
        if data_centroid_labels is None:
            raise ValueError('data_centroid_labels cannot be None')
        if data_centroid_labels.ndim != 1:
            raise ValueError('data_centroid_labels must be a 1D numpy array')
        if prev_centroids is None:
            raise ValueError('prev_centroids cannot be None')
        if prev_centroids.ndim != 2:
            raise ValueError('prev_centroids must be a 2D numpy array')
        if prev_centroids.shape[0] != k:
            raise ValueError('prev_centroids must have k rows')

        new_centroids = np.ndarray(shape=(k, data_.shape[1]))
        for i in range(k):
            temp_arr = np.where(data_centroid_labels == i, True, False)
            if not np.any(temp_arr):
                new_centroids[i] = data_[np.random.choice(data_.shape[0], 1)]
                continue
            new_centroids[i] = np.mean(data_[temp_arr], axis=0)
        return new_centroids, new_centroids - prev_centroids

    def update_centroids(self, k: int, data_centroid_labels: np.ndarray = None,
                         prev_centroids: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
            Computes each of the K centroids (means) based on the data assigned to each cluster

            :param k: int.
                Number of clusters
            :param data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
                Holds index of the assigned cluster of each data sample
            :param prev_centroids. ndarray. shape=(k, self.num_features)
                Holds centroids for each cluster computed on the PREVIOUS time step

            :return new_centroids: ndarray. shape=(k, self.num_features).
                Centroids for each cluster computed on the CURRENT time step
            :return centroid_diff. ndarray. shape=(k, self.num_features).
                Difference between current and previous centroid values

            NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
            i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
                For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
            In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
            randomly selected from the dataset.
            """
        if data_centroid_labels is None:
            data_centroid_labels = self.data_centroid_labels
            if data_centroid_labels is None:
                raise ValueError('data_centroid_labels cannot be None')
        if prev_centroids is None:
            prev_centroids = self.centroids
            if prev_centroids is None:
                raise ValueError('prev_centroids cannot be None')
        return KMeans.update_centroids_static(self.data, k, data_centroid_labels, prev_centroids)

    @staticmethod
    def compute_inertia_static(data_: np.ndarray, centroids: np.ndarray, data_centroid_labels: np.ndarray) -> float:
        """
        Mean squared distance between every data sample and its assigned (nearest) centroid

        :param data_: ndarray. shape=(num_samps, num_features).
            The dataset to be clustered
        :param centroids: ndarray. shape=(k, self.num_features).
            Current centroids for the k clusters.
        :param data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample

        :return inertia: float.
            The average squared distance between every data sample and its assigned cluster centroid.
        """
        if data_ is None:
            raise ValueError('data_ cannot be None')
        if data_.ndim != 2:
            raise ValueError('data_ must be a 2D numpy array')
        if centroids is None:
            raise ValueError('centroids cannot be None')
        if centroids.ndim != 2:
            raise ValueError('centroids must be a 2D numpy array')
        if data_centroid_labels is None:
            raise ValueError('data_centroid_labels cannot be None')
        if data_centroid_labels.ndim != 1:
            raise ValueError('data_centroid_labels must be a 1D numpy array')
        if data_centroid_labels.shape[0] != data_.shape[0]:
            raise ValueError('data_centroid_labels must have the same number of samples as data_')
        return np.mean(np.square(
            KMeans.dist_pts_to_pts(data_, centroids)[np.arange(data_.shape[0]), data_centroid_labels]))

    def compute_inertia(self) -> float:
        """
        Mean squared distance between every data sample and its assigned (nearest) centroid

        :return inertia: float.
            The average squared distance between every data sample and its assigned cluster centroid.
        """
        return KMeans.compute_inertia_static(self.data, self.centroids, self.data_centroid_labels)


    @staticmethod
    def plot_clusters_static(data_: np.ndarray, labels: np.ndarray, k: int, x_label: str = None,
                             y_label: str = None, title: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Creates a scatter plot of the data color-coded by cluster assignment.

        :param data_: ndarray. shape=(num_samps, num_features).
            The dataset to be plotted.
        :param labels: ndarray of ints. shape=(num_samps,).
            Holds index of the assigned cluster of each data sample
        :param k: int.
            Number of clusters
        :param x_label: str.
            Label for the x-axis
        :param y_label: str.
            Label for the y-axis
        :param title: str.
            Title for the plot

        :return fig: matplotlib.figure.Figure.
            The figure object for the plot
        :return ax: matplotlib.axes.Axes.

        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). To make sure you change your colors to be clearly differentiable,
        use either the Okabe & Ito or one of the Petroff color palettes: https://github.com/proplot-dev/proplot/issues/424
        Each string in the `colors` list that starts with # is the hexadecimal representation of a color (blue, red, etc.)
        that can be passed into the color `c` keyword argument of plt.plot or plt.scatter.
            Pick one of the palettes with a generous number of colors so that you don't run out if k is large (e.g. >6).
        """
        colors = ["#004949", "#FF6DB6", "#490092", "#B66DFF", "#B6DBFF", "#924900", "#24FF24",
                  "#009292", "#FFB6DB", "#006DDB", "#6DB6FF", "#920000", "#DB6D00", "#FFFF6D"]
        if data_ is None:
            raise ValueError('data_ cannot be None')
        if data_.ndim != 2:
            raise ValueError('data_ must be a 2D numpy array')
        if data_.shape[1] != 2:
            raise ValueError('data_ must have 2 features')
        if labels is None:
            raise ValueError('labels cannot be None')
        if labels.ndim != 1:
            raise ValueError('labels must be a 1D numpy array')
        if labels.shape[0] != data_.shape[0]:
            raise ValueError('labels must have the same number of samples as data_')
        if centroids is None:
            raise ValueError('centroids cannot be None')
        if centroids.ndim != 2:
            raise ValueError('centroids must be a 2D numpy array')
        if centroids.shape[0] != k:
            raise ValueError('centroids must have k rows')
        if x_label is None:
            x_label = 'X'
        if y_label is None:
            y_label = 'Y'
        if title is None:
            title = 'K-Means Clustering'

        fig = plt.figure(figsize=(8, 5))
        ax = fig.subplots()
        for i in range(k):
            temp = data_[labels == i]
            ax.scatter(temp[:, 0], temp[:, 1], color=colors[i], label=f'Cluster {i}')
            ax.scatter(centroids[i, 0], centroids[i, 1], color='black', marker='x', s=100)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        xMin, xMax = data_[:, 0].min(), data_[:, 0].max()
        yMin, yMax = data_[:, 1].min(), data_[:, 1].max()
        ax.set_xlim(xMin * 1.05, xMax * 1.05)
        ax.set_ylim(yMin * 1.05, yMax * 1.05)
        ax.set_xticks(np.linspace(xMin, xMax, 5))
        ax.set_yticks(np.linspace(yMin, yMax, 5))
        ax.set_xticklabels([f'{i:.2f}' for i in np.linspace(xMin, xMax, 5)])
        ax.set_yticklabels([f'{i:.2f}' for i in np.linspace(yMin, yMax, 5)])
        legend = fig.legend(title="Clusters", bbox_to_anchor=(1.2, 1.01), loc="upper right")
        legend._legend_box.align = "left"
        fig.subplots_adjust(bottom=-0.25)
        fig.tight_layout()
        return fig, ax


    def plot_clusters(self, x_label: str = None, y_label: str = None, title: str = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Creates a scatter plot of the data color-coded by cluster assignment.

        :param x_label: str.
            Label for the x-axis
        :param y_label: str.
            Label for the y-axis
        :param title: str.
            Title for the plot

        :return fig, ax: Tuple[plt.Figure, plt.Axes]
            The matplotlib figure and axes objects.

        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). To make sure you change your colors to be clearly differentiable,
        use either the Okabe & Ito or one of the Petroff color palettes:
        """
        return KMeans.plot_clusters_static(self.data, self.data_centroid_labels, self.centroids, self.k, x_label, y_label)


    @staticmethod
    def elbow_plot_static(data_: np.ndarray, max_k: int, n_iter: int = 1) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """
        Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        :param data_: np.ndarray. shape=(num_samps, num_features).
            The dataset to be clustered
        :param max_k: int.
            Run k-means with k=1,2,...,max_k.
        :param n_iter: int.
            Number of times to run K-means with the designated `k` value.

        :return fig, (ax1, ax2): Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]
            The matplotlib figure and axes objects.
        """
        colors = ["#004949", "#FF6DB6", "#490092", "#B66DFF", "#B6DBFF", "#924900", "#24FF24",
                  "#009292", "#FFB6DB", "#006DDB", "#6DB6FF", "#920000", "#DB6D00", "#FFFF6D"]
        if data_ is None:
            raise ValueError('data_ cannot be None')
        if data_.ndim != 2:
            raise ValueError('data_ must be a 2D numpy array')
        if data_.shape[1] != 2:
            raise ValueError('data_ must have 2 features')
        if not isinstance(max_k, int):
            raise ValueError('max_k must be an integer')
        if max_k < 1:
            raise ValueError('max_k must be greater than or equal to 1')

        inertia = []
        inertia_diff = [0]
        for i in range(1, max_k + 1):
            inertia.append(KMeans.cluster_batch_static(data_, i, n_iter)[1])
            if i > 1:
                inertia_diff.append(inertia[-1] - inertia[-2])
        fig = plt.figure(figsize=(8, 5))
        axes = (ax1, ax2) = fig.subplots(1, 2)
        ax1.plot(range(1, max_k + 1), inertia, marker='o', c=colors[0])
        ax2.plot(range(1, max_k + 1), inertia_diff, marker='o', c=colors[1])
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Inertia Difference')
        ax1.set_xticks(range(1, max_k + 1))
        ax2.set_xticks(range(1, max_k + 1))
        ax1.set_title('Elbow Plot')
        ax2.set_title('Inertia Difference')
        fig.tight_layout()
        return fig, axes

    def elbow_plot(self, max_k, n_iter: int = 1):
        """
        Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        :param max_k: int.
            Run k-means with k=1,2,...,max_k.
        :param n_iter: int.
            Number of times to run K-means with the designated `k` value.

        :return fig, ax: Tuple[plt.Figure, plt.Axes]
            The matplotlib figure and axes objects.
        """
        return KMeans.elbow_plot_static(self.data, max_k, n_iter=n_iter)

    @staticmethod
    def replace_color_with_centroid_static(data_: np.ndarray, centroids: np.ndarray, data_labels: np.ndarray) -> np.ndarray:
        """Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        :param data_: ndarray. shape=(num_samps, num_features).
            The dataset to be clustered
        :param centroids: ndarray. shape=(k, self.num_features).
            Centroids for each cluster
        :param data_labels: ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        :return None
        """
        if data_ is None:
            raise ValueError('data_ cannot be None')
        if data_.ndim != 2:
            raise ValueError('data_ must be a 2D numpy array')
        if data_.shape[1] != 3:
            raise ValueError('data_ must have 3 features')
        if centroids is None:
            raise ValueError('centroids cannot be None')
        if centroids.ndim != 2:
            raise ValueError('centroids must be a 2D numpy array')
        if centroids.shape[1] != 3:
            raise ValueError('centroids must have 3 features')
        data_copy = data_.copy()
        for i in range(data_.shape[0]):
            label = data_labels[i]
            data_copy[i] = centroids[label]
        return data_copy

    def replace_color_with_centroid(self):
        """Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.
        """
        self.set_data(KMeans.replace_color_with_centroid_static(self.data, self.centroids, self.data_centroid_labels))


if __name__ == '__main__':
    # Tests
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    centroids = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])


    # dist_pts_to_pts
    print(KMeans.dist_pts_to_pts(data, centroids))
    print(KMeans.dist_pts_to_pts(data, centroids, analysis.Analysis.l1_norm))
    print(KMeans.dist_pts_to_pts(data, centroids, analysis.Analysis.l2_norm))
    print(KMeans.dist_pts_to_pts(data, centroids, analysis.Analysis.lp_norm))
    print(KMeans.dist_pts_to_pts(data, centroids, analysis.Analysis.l_inf_norm))
    print(KMeans.dist_pts_to_pts(data, centroids, analysis.Analysis.l1_norm, 2))
    print(KMeans.dist_pts_to_pts(data, centroids, analysis.Analysis.l2_norm, 2))
    print(KMeans.dist_pts_to_pts(data, centroids, analysis.Analysis.lp_norm, 2))
    print(KMeans.dist_pts_to_pts(data, centroids, analysis.Analysis.l_inf_norm, 2))
    print(KMeans.dist_pts_to_pts(data, centroids, analysis.Analysis.lp_norm, 3))




