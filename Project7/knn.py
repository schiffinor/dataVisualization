"""
knn.py
K-Nearest Neighbors algorithm for classification
Roman Schiffino
CS 251: Data Analysis and Visualization
Spring 2024
"""
import warnings

import scipy as sp
from numpy.typing import ArrayLike

import data_transformations
from analysis import Analysis
from typing import List, NewType, Tuple, Callable
from kmeans import KMeans as kM

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from classifier import Classifier



# Define a type for a function that takes two np.ndarrays and returns a scalar
Norm = NewType("Norm", Callable[[np.ndarray, np.ndarray, ...], float])

# Aliases for norming functions in KMeans
v_norm = kM.norm_validation
ndnc = kM.ndim_norm_choice_and_call


def d_pt_to_pts(pt: np.ndarray | ArrayLike, pts: np.ndarray | ArrayLike, norm: Norm = Analysis.l2_norm, normP: float | int = 2.0) -> np.ndarray:
    """
    Compute the Euclidean distance between data sample `pt` and all the pts in `pts`.

    :param pt: ndarray. shape=(num_features,)
        The point to compute the distance from the exemplars.
    :param pts: ndarray. shape=(C, num_features)
        C pts, where C is an int.
    :param norm: Norm.
        A norm function to call.
    :param normP: float.
        The p value for the norm function.

    :return distances: ndarray. shape=(C,).
        distance between pt and each of the C Exemplars in `pts`.
    """
    if pt is None:
        raise ValueError('pt cannot be None')
    if pts is None:
        raise ValueError('pts cannot be None')
    if not isinstance(pt, np.ndarray):
        if not hasattr(pt, "__array__"):
            raise ValueError("pt must be an array-like object")
        pt = np.array(pt)
    if not isinstance(pts, np.ndarray):
        if not hasattr(pts, "__array__"):
            raise ValueError("pts must be an array-like object")
        pts = np.array(pts)
    if isinstance(pt, np.ndarray):
        pt_np = pt
    else:
        pt_np = np.array(pt)
    if isinstance(pts, np.ndarray):
        pts_np = pts
    else:
        pts_np = np.array(pts)
    if pt.ndim != 1:
        raise ValueError('pt must be a 1D numpy array')
    if pts.ndim != 2:
        raise ValueError('pts must be a 2D numpy array')
    if pts.shape[1] != pt.shape[0]:
        raise ValueError('pts and pt must have the same number of features')
    v_norm(norm, normP)

    pt_arr = np.vstack([pt_np for _ in range(pts.shape[0])])
    return ndnc(pt_arr, pts_np, norm, normP)


def d_pts_to_pts(pts1: np.ndarray | ArrayLike, pts2: np.ndarray | ArrayLike, norm: Norm = Analysis.l2_norm, normP: float | int = 2.0) -> np.ndarray:
    """
    Compute the Euclidean distance between each point in `pts1` and all the points in `pts2`.

    :param pts1: ndarray. shape=(A, num_features)
        A pts, where A is an int.
    :param pts2: ndarray. shape=(B, num_features)
        B pts, where B is an int.
    :param norm: Norm.
        A norm function to call.
    :param normP: float.
        The p value for the norm function.

    :return distances: ndarray. shape=(A, B).
        distance between each pt in pts1 and each pt in pts2.
    """
    if pts1 is None:
        raise ValueError('pts1 cannot be None')
    if pts2 is None:
        raise ValueError('pts2 cannot be None')
    if not isinstance(pts1, np.ndarray):
        if not hasattr(pts1, "__array__"):
            raise ValueError("y must be an array-like object")
        pts1 = np.array(pts1)
    if not isinstance(pts2, np.ndarray):
        if not hasattr(pts2, "__array__"):
            raise ValueError("y_pred must be an array-like object")
        pts2 = np.array(pts2)
    if not hasattr(pts1, "__array__"):
        raise ValueError('pts1 must be an array-like object')
    if not hasattr(pts2, "__array__"):
        raise ValueError('pts2 must be an array-like object')
    if not isinstance(pts1, np.ndarray):
        pts1 = np.array(pts1)
    if not isinstance(pts2, np.ndarray):
        pts2 = np.array(pts2)
    if pts1.ndim != 2:
        raise ValueError('pts1 must be a 2D numpy array')
    if pts2.ndim != 2:
        raise ValueError('pts2 must be a 2D numpy array')
    if pts1.shape[1] != pts2.shape[1]:
        raise ValueError('pts1 and pts2 must have the same number of features')
    v_norm(norm, normP)

    pts1_arr_list = [pts1[i] for i in range(pts1.shape[0]) for _ in range(pts2.shape[0])]
    pts1_arr = np.vstack(pts1_arr_list)
    pts2_arr = np.vstack([pts2 for _ in range(pts1.shape[0])])
    return ndnc(pts1_arr, pts2_arr, norm, normP).reshape(pts1.shape[0], pts2.shape[0])


class KNN(Classifier):
    """
    K-Nearest Neighbors supervised learning algorithm
    """

    def __init__(self, num_classes: int):
        """
        KNN constructor

        :param num_classes: int
            Number of classes in the classification problem
        """
        super().__init__(num_classes)

        # exemplars: ndarray. shape=(num_train_samps, num_features).
        #   Memorized training examples
        self.exemplars = None
        # classes: ndarray. shape=(num_train_samps,).
        #   Classes of memorized training examples
        self.classes = None

    def train(self, data: np.ndarray, y: np.ndarray):
        """
        Train the KNN classifier on the data `data`, where training samples have corresponding
        class labels in `y`.

        :param data: ndarray. shape=(num_train_samps, num_features). Data to learn / train on.
        :param y: ndarray. shape=(num_train_samps,). Corresponding class of each data sample.
        """
        if not isinstance(data, np.ndarray):
            if not hasattr(data, "__array__"):
                warnings.warn("data must be an array-like object. Will force convert.")
                data = np.array(data)
        if not isinstance(y, np.ndarray):
            if not hasattr(y, "__array__"):
                warnings.warn("y should be an array-like object. Will force convert.")
                y = np.array(y)
        if len(data) != len(y):
            raise ValueError("data and y must have the same length")
        self.exemplars = data
        self.classes = y

    def predict(self, data: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.

        :param data: np.ndarray. shape=(num_test_samps, num_features). 
            Data to predict the class of
            Need not be the data used to train the network.
        :param k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.

        :return np.ndarray of nonnegative ints. shape=(num_test_samps,). Predicted class of each test data
        sample.
        """
        return self.predict_static(data, self.exemplars, self.classes, k)

    def predict_mod(self, data: np.ndarray, k: int = 1, norm: Norm = Analysis.l2_norm, normP: float | int = 2.0) -> np.ndarray:
        """
        Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.

        :param data: np.ndarray. shape=(num_test_samps, num_features).
            Data to predict the class of
            Need not be the data used to train the network.
        :param k: int.
            Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.
        :param norm: Norm.
            Function to compute the distance between two data samples.
        :param normP: float.
            The p-value for the Minkowski-lp distance metric. Default is 2 for Euclidean distance.

        :return np.ndarray of nonnegative ints. shape=(num_test_samps,). Predicted class of each test data
        sample.
        """
        return self.predict_static(data, self.exemplars, self.classes, k, norm, normP)

    @staticmethod
    def predict_static(data: np.ndarray, exemplars: np.ndarray, classes: np.ndarray, k: int = 1,
                       norm: Norm = Analysis.l2_norm, normP: float | int = 2.0) -> np.ndarray:
        """
        Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.
        
        :param data: np.ndarray. shape=(num_test_samps, num_features).
            Data to predict the class of
        :param exemplars: np.ndarray. shape=(num_train_samps, num_features).
            Memorized training examples
        :param classes: np.ndarray. shape=(num_train_samps,).
            Classes of memorized training examples
        :param k: int.
            Determines the neighborhood size of training points around each test sample used to
        :param norm: Norm.
            Function to compute the distance between two data samples.
        :param normP: float.
            The p-value for the Minkowski-lp distance metric. Default is 2 for Euclidean distance.
        :return: np.ndarray. d_type=int shape=(num_test_samps,).
            Predicted class of each test data sample.

        - Compute the distance from each test sample to all the training exemplars.
        - Among the closest `k` training exemplars to each test sample, count up how many belong
        to which class.
        - The predicted class of the test sample is the majority vote.
        """
        if not isinstance(data, np.ndarray):
            if not hasattr(data, "__array__"):
                warnings.warn("data must be an array-like object. Will force convert.")
                data = np.array(data)
        if not isinstance(exemplars, np.ndarray):
            if not hasattr(exemplars, "__array__"):
                warnings.warn("exemplars must be an array-like object. Will force convert.")
                exemplars = np.array(exemplars)
        if not isinstance(classes, np.ndarray):
            if not hasattr(classes, "__array__"):
                warnings.warn("classes must be an array-like object. Will force convert.")
                classes = np.array(classes)
        if not hasattr(norm, "__call__"):
            raise ValueError("norm must be a callable function")
        if norm.__code__.co_argcount < 2:
            raise ValueError("norm must take at least 2 arguments")
        if k < 1:
            raise ValueError("k must be a positive integer")
        if exemplars is None or classes is None:
            raise ValueError("KNN must be trained before making predictions")
        if data.shape[1] != exemplars.shape[1]:
            raise ValueError("data and exemplars must have the same number of features")
        if k > len(exemplars):
            raise ValueError("k must be less than the number of training exemplars")
        if normP < 0:
            raise ValueError("normP must be non-negative")
        if len(data) == 0:
            raise ValueError("data is empty")
        if len(exemplars) == 0:
            raise ValueError("exemplars is empty")
        if len(classes) == 0:
            raise ValueError("classes is empty")
        if len(exemplars) != len(classes):
            raise ValueError("exemplars and classes must have the same length")

        # Compute the distance from each test sample to all the training exemplars.
        if data.shape[0] * exemplars.shape[0] * data.shape[1] < 1e7:
            dists = d_pts_to_pts(data, exemplars, norm, normP)
        else:
            dists = np.array([d_pt_to_pts(data[i], exemplars, norm, normP) for i in range(data.shape[0])])
        classes = classes.astype(int)[:, np.newaxis].T
        class_arr = np.broadcast_to(classes, (dists.shape[0], classes.shape[1]))
        k_nearest_sort = dists.argsort(axis=1)
        k_nearest_sort = k_nearest_sort[:, :k]
        k_nearest_classes = np.take_along_axis(class_arr, k_nearest_sort, axis=1)
        return sp.stats.mode(k_nearest_classes, axis=1)[0]

    def plot_predictions(self, k, n_sample_pts, norm: Norm = Analysis.l2_norm, normP: float | int = 2.0):
        """
        Paints the data space in colors corresponding to which class the classifier would
         hypothetically assign to data samples appearing in each region.

        :param k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.
        :param n_sample_pts: int.
            How many points to divide up the input data space into along the x and y axes to plug
            into KNN at which we are determining the predicted class. Think of this as regularly
            spaced 2D "fake data" that we generate and plug into KNN and get predictions at.
        :param norm: Norm.
            Function to compute the distance between two data samples.
        :param normP: float.
            The p-value for the Minkowski-lp distance metric. Default is 2 for Euclidean distance.
        :return: fig, ax.
            matplotlib figure and axis objects.

        - Pick a discrete/qualitative color scheme. We suggest, like in the clustering project, to
        use either the Okabe & Ito or one of the Petroff color palettes: https://github.com/proplot-dev/proplot/issues/424
        - Wrap your colors list as a `ListedColormap` object (already imported above) so that matplotlib can parse it.
        - Make an ndarray of length `n_sample_pts` of regularly spaced points between -40 and +40.
        - Call `np.meshgrid` on your sampling vector to get the x and y coordinates of your 2D
        "fake data" sample points in the square region from [-40, 40] to [40, 40].
            - Example: x, y = np.meshgrid(samp_vec, samp_vec)
        - Combine your `x` and `y` sample coordinates into a single ndarray and reshape it so that
        you can plug it in as your `data` in self.predict.
            - Shape of `x` should be (n_sample_pts, n_sample_pts). You want to make your input to
            self.predict of shape=(n_sample_pts*n_sample_pts, 2).
        - Reshape the predicted classes (`y_pred`) in a square grid format for plotting in 2D.
        shape=(n_sample_pts, n_sample_pts).
        - Use the `plt.pcolormesh` function to create your plot. Use the `cmap` optional parameter
        to specify your discrete ColorBrewer color palette.
        - Add a colorbar to your plot
        """
        # Custom Color Maps
        colors = ["#004949", "#FF6DB6", "#490092", "#B66DFF", "#B6DBFF", "#924900", "#24FF24",
                  "#009292", "#FFB6DB", "#006DDB", "#6DB6FF", "#920000", "#DB6D00", "#FFFF6D"]
        light_colors = colors[::2]
        dark_colors = list(map(lambda hexCode: "#" + "".join(map(lambda col: ("0x%0.2X" % int((255 + int(col, 16)) / 2))[2:],
                                                                 [hexCode[1][1 + 2 * i: 3 + 2 * i] for i in range(3)])),
                               enumerate(light_colors)))
        custom_cmap_light = ListedColormap(light_colors)
        custom_cmap_dark = ListedColormap(dark_colors)

        # Generate fake data
        samp_vec = np.linspace(-40, 40, n_sample_pts)
        x, y = np.meshgrid(samp_vec, samp_vec)
        data = np.vstack([x.flatten(), y.flatten()]).T
        y_pred = self.predict_mod(data, k, norm, normP)
        y_pred = y_pred.reshape(n_sample_pts, n_sample_pts)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.pcolormesh(x, y, y_pred, cmap=custom_cmap_light)
        ax.set_title(f"KNN Predictions with k={k}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap_light), ax=ax)
        return fig, ax

    def fullPlot(self, data: np.ndarray, y: np.ndarray, k: int = 1, n_sample_pts: int = 100,
                 norm: Norm = Analysis.l2_norm, normP: float | int = 2.0):
        """
        Paints the data space in colors corresponding to which class the classifier would
         hypothetically assign to data samples appearing in each region.

        :param norm:
        :param normP:
        :param norm:
        :param data: np.ndarray. shape=(num_train_samps, num_features). Data to learn / train on.
        :param y: np.ndarray. shape=(num_train_samps,). Corresponding class of each data sample.
        :param k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.
        :param n_sample_pts: int.
            How many points to divide up the input data space into along the x and y axes to plug
            into KNN at which we are determining the predicted class.
        :return: fig, ax.
            matplotlib figure and axis objects.
        """
        # Data validation
        if not isinstance(data, np.ndarray):
            if not hasattr(data, "__array__"):
                warnings.warn("data must be an array-like object. Will force convert.")
                data = np.array(data)
        if not isinstance(y, np.ndarray):
            if not hasattr(y, "__array__"):
                warnings.warn("y should be an array-like object. Will force convert.")
                y = np.array(y)
        if len(data) != len(y):
            raise ValueError("data and y must have the same length")
        if not isinstance(k, int):
            raise ValueError("k must be an integer")
        if k < 1:
            raise ValueError("k must be a positive integer")
        if not isinstance(n_sample_pts, int):
            raise ValueError("n_sample_pts must be an integer")
        if n_sample_pts < 1:
            raise ValueError("n_sample_pts must be a positive integer")

        # Custom Color Maps
        colors = ["#004949", "#FF6DB6", "#490092", "#B66DFF", "#B6DBFF", "#924900", "#24FF24",
                  "#009292", "#FFB6DB", "#006DDB", "#6DB6FF", "#920000", "#DB6D00", "#FFFF6D"]
        light_colors = colors[::2]
        dark_colors = list(
            map(lambda hexCode: "#" + "".join(map(lambda col: ("0x%0.2X" % int((255 + int(col, 16)) / 2))[2:],
                                                  [hexCode[1][1 + 2 * i: 3 + 2 * i] for i in range(3)])),
                enumerate(light_colors)))
        custom_cmap_light = ListedColormap(light_colors)
        custom_cmap_dark = ListedColormap(dark_colors)

        # Data Stats
        num_classes = len(np.unique(y))
        num_features = data.shape[1]

        # Transform the data
        data = data_transformations.normalize(data)
        data = data_transformations.center(data)
        data = 80 * data

        # Train the model
        self.train(data, y)

        # Generate fake data
        samp_vec = np.linspace(-40, 40, n_sample_pts)
        x, y = np.meshgrid(samp_vec, samp_vec)
        data = np.vstack([x.flatten(), y.flatten()]).T
        y_pred = self.predict(data, k)
        y_pred = y_pred.reshape(n_sample_pts, n_sample_pts)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.pcolormesh(x, y, y_pred, cmap=custom_cmap_light)
        ax.set_title(f"KNN Predictions with k={k}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap_light), ax=ax)
        # Plot the data
        for i in range(num_classes):
            ax.scatter(data[y == i, 0], data[y == i, 1], c=y[i], cmap=custom_cmap_dark, label=f"class {i}", s=30)
        ax.legend()
        xMin, xMax = -40, 40
        yMin, yMax = -40, 40
        ax.set_xlim(xMin * 1.05, xMax * 1.05)
        ax.set_ylim(yMin * 1.05, yMax * 1.05)
        ax.set_xticks(np.linspace(xMin, xMax, 5))
        ax.set_yticks(np.linspace(yMin, yMax, 5))
        ax.set_xticklabels([f'{i:.2f}' for i in np.linspace(xMin, xMax, 5)])
        ax.set_yticklabels([f'{i:.2f}' for i in np.linspace(yMin, yMax, 5)])
        legend = fig.legend(title="Clusters", bbox_to_anchor=(1.2, 1.01), loc="upper right")
        legend._legend_box.align = "left"

        fig.tight_layout()
        return fig, ax



