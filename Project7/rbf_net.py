"""rbf_net.py
Radial Basis Function Neural Network
Roman Schiffino
CS 251: Data Analysis Visualization
Spring 2023
"""
import numpy as np
from typing import Tuple, List, Union, Dict, NewType, Callable

import analysis
import kmeans
import analysis as an
import warnings as wn

import setNorm

# Define a type for a function that takes two np.ndarrays and returns a scalar
Norm = NewType("Norm", Callable[[np.ndarray, np.ndarray, ...], float])

Norm_ndim = NewType("Norm_ndim", Callable[[np.ndarray, np.ndarray, ...], np.ndarray])

Norm_plist = NewType("Norm_plist", Callable[[np.ndarray, np.ndarray, ...], np.ndarray])


class RBF_Net:
    def __init__(self, num_hidden_units: int, num_classes: int, norm: Norm = an.Analysis.l2_norm,
                 normP: int | float = 2, norm_ndim: Norm_ndim = an.Analysis.l2_norm_ndim,
                 norm_plist: Norm_plist = an.Analysis.lp_norm_v2_pList):
        """
        RBF network constructor

        :param num_hidden_units: int.
            Number of hidden units in network.
            NOTE: does NOT include bias unit
        :param num_classes: int.
            Number of output units in network.
            Equals number of possible classes in dataset

        TODO:
        - Define number of hidden units as an instance variable called `k` (as in k clusters)
            (You can think of each hidden unit as being positioned at a cluster center)
        - Define number of classes (number of output units in network) as an instance variable
        """
        # k: Number of hidden units in network. Does NOT include bias unit
        self.k = num_hidden_units
        # num_classes: Number of output units in network. Equals number of possible classes in dataset
        self.class_count = num_classes
        # norm: Function to compute the distance between two vectors
        # Norm function is actually a metric, must take two 1-dim np.ndarrays and return a scalar.
        #   Default is the L2 norm
        # Each Norm should have an associated multi-dimensional norm function and an associated point-list norm function.
        #   The multi-dimensional norm function should take two n-dimensional np.ndarrays and return an
        #   (n-1)-dimensional norm.
        #   The point-list norm function should take two lists (of length i and j)
        #   of n-dimensional points and return an i*j array of scalar distances.
        #   The point-list norm function should be used for the k-means algorithm.
        #   Custom Norms and their associations can be registered through

        self.normClass = None
        self.normType = None
        self.normP = None

        self.norm = None
        self.normNdim = None
        self.normPlist = None
        self.dist = None
        self.distNdim = None
        self.distPlist = None
        self.setNorm(norm, norm_ndim, norm_plist, normP)

        # prototypes: Hidden unit prototypes (i.e. center)
        #   shape=(num_hidden_units, num_features)
        self.prototypes = None
        # sigmas: Hidden unit sigmas: controls how active each hidden unit becomes to inputs that
        # are similar to the unit's prototype (i.e. center).
        #   shape=(num_hidden_units,)
        #   Larger sigma -> hidden unit becomes active to dissimilar inputs
        #   Smaller sigma -> hidden unit only becomes active to similar inputs
        self.sigmas = None
        # wts: Weights connecting hidden and output layer neurons.
        #   shape=(num_hidden_units+1, num_classes)
        #   The reason for the +1 is to account for the bias (a hidden unit whose activation is always
        #   set to 1).
        self.wts = None

    def setNorm(self, norm: Norm, norm_ndim: Norm_ndim = None, norm_plist: Norm_plist = None,
                normP: int | float = None):
        self.normClass = setNorm.setNorm(norm, norm_ndim, norm_plist, normP)
        self.normType = self.normClass.getNorm()
        self.normP = self.normClass.getNormP()

        self.norm = self.normClass.norm
        self.normNdim = self.normClass.ndim_norm
        self.normPlist = self.normClass.plist_norm
        self.dist = self.normClass.pt_dist
        self.distNdim = self.normClass.ndim_dist
        self.distPlist = self.normClass.plist_dist

    def get_prototypes(self):
        """
        Returns the hidden layer prototypes (centers)

        (Should not require any changes)

        :returns: ndarray. shape=(k, num_features).
            Hidden layer prototypes
        """
        return self.prototypes

    def get_num_hidden_units(self):
        """
        Returns the number of hidden layer prototypes (centers/"hidden units").

        :returns: int, Number of hidden units.
        """
        return self.k

    def get_num_output_units(self):
        """
        Returns the number of output layer units.


        :returns: Int, Number of output units
        """
        return self.class_count

    def avg_cluster_dist(self, data: np.ndarray, centroids: np.ndarray, cluster_assignments: np.ndarray):
        """
        Compute the average distance between each cluster center and data points that are
        assigned to it.

        :param data: ndarray. shape=(num_samps, num_features).
            Data samples
        :param centroids: ndarray. shape=(num_clusters, num_features).
            Cluster centers
        :param cluster_assignments: ndarray. shape=(num_samps,).
            Cluster assignment for each data sample

        :returns: ndarray. shape=(num_clusters,).
            Average distance between each cluster center and data points that are assigned to it.
        """
        return np.array([np.einsum("i->", self.distPlist(data[cluster_assignments == i], centroids[i])) / np.einsum(
            "i->", cluster_assignments == i) for i in range(centroids.shape[0])])

    def initialize(self, data):
        """Initialize hidden unit centers using K-means clustering and initialize sigmas using the
        average distance within each cluster

        :param data: ndarray, shape=(num_samps, num_features).
            Data to initialize the network


        TODO:
        - Determine `self.prototypes` (see constructor for shape). Prototypes are the centroids
        returned by K-means. It is recommended to use the 'batch' version of K-means to reduce the
        chance of getting poor initial centroids.
            - To increase the chance that you pick good centroids, set the parameter controlling the
            number of iterations > 1 (e.g. 5)
        - Determine self.sigmas as the average distance between each cluster center and data points
        that are assigned to it. Hint: You implemented a method to do this!
        """
        pass

    def linear_regression(self, A, y):
        """Performs linear regression
        CS251: Adapt your SciPy lstsq code from the linear regression project.
        CS252: Adapt your QR-based linear regression solver

        :param A: ndarray, shape=(num_samps, num_features).
            Design matrix
        :param y: ndarray, shape=(num_samps,).
            Target values

        :returns: ndarray. shape=(num_features,).
            Weights that minimize the squared error
        NOTE: Remember to handle the intercept ("homogenous coordinate")
        """

    def hidden_act(self, data):
        """
        Compute the activation of the hidden layer units

        :returns: ndarray, shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation
        """
        pass

    def output_act(self, hidden_acts):
        """Compute the activation of the output layer units

        :param hidden_acts: ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.

        :returns: ndarray. shape=(num_samps, num_output_units).
            Activation of each unit in the output layer to each of the data samples.

        NOTE:
        - Assumes that learning has already taken place
        - Can be done without any for loops.
        - Don't forget about the bias unit!
        """
        pass

    def train(self, data, y):
        """
        Train the radial basis function network


        :param data: ndarray, shape=(num_samps, num_features).
            Data to learn / train on.
        :param y: ndarray, shape=(num_samps, ).
            Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the correct classes of each training sample. To solve for the weights going FROM all of the
        hidden units TO output unit c, recode the class vector `y` to 1s and 0s:
            1 if the class of a data sample in `y` is c
            0 if the class of a data sample in `y` is not c

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.
        """
        pass

    def predict(self, data):
        """Classify each sample in `data`

        :param data: ndarray, shape=(num_samps, num_features).
            Data to predict classes for.
            Need not be the data used to train the network

        :returns: ndarray of nonnegative ints, shape=(num_samps,).
            Predicted class of each data sample.

        TODO:
        - Pass the data thru the network (input layer -> hidden layer -> output layer).
        - For each data sample, the assigned class is the index of the output unit that produced the
        largest activation.
        """
        pass

    def accuracy(self, y, y_pred):
        """Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        :param y: ndarray, shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        :param y_pred: ndarray, shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        :returns: float, Between 0 and 1.
         Proportion correct classification.

        NOTE: Can be done without any loops
        """
        pass
