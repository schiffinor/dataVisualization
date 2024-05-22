"""classifer.py
Generic classifier data type
YOUR NAME HERE
CS 251: Data Analysis and Visualization
Spring 2024
"""
import warnings

import numpy as np
from numpy.typing import ArrayLike


class Classifier:
    """Parent class for classifiers"""

    def __init__(self, num_classes):
        """
        Classifier constructor

        :param num_classes: int
            Number of classes in the classification problem
        """
        self.num_classes = num_classes

    @staticmethod
    def accuracy(y: np.ndarray | ArrayLike, y_pred: np.ndarray | ArrayLike):
        """
        Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        :param y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        :param y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        """
        if not isinstance(y, np.ndarray):
            if not hasattr(y, "__array__"):
                raise ValueError("y must be an array-like object")
            y = np.array(y)
        if not isinstance(y_pred, np.ndarray):
            if not hasattr(y_pred, "__array__"):
                raise ValueError("y_pred must be an array-like object")
            y_pred = np.array(y_pred)
        if len(y) == 0:
            warnings.warn("y is empty, 0 returned as default accuracy")
            return 0
        if len(y) != len(y_pred):
            raise ValueError("y and y_pred must have the same length")
        return np.sum(y == y_pred) / float(len(y))

    def confusion_matrix(self, y: np.ndarray | ArrayLike, y_pred: np.ndarray | ArrayLike):
        """Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        :param y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        :param y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        """
        if not isinstance(y, np.ndarray):
            if not hasattr(y, "__array__"):
                raise ValueError("y must be an array-like object")
            y = np.array(y)
        if not isinstance(y_pred, np.ndarray):
            if not hasattr(y_pred, "__array__"):
                raise ValueError("y_pred must be an array-like object")
            y_pred = np.array(y_pred)
        if len(y) == 0:
            warnings.warn("y is empty, empty array returned as default confusion matrix")
            return np.array([])
        if len(y) != len(y_pred):
            raise ValueError("y and y_pred must have the same length")
        return np.array([[np.sum((y == i) and (y_pred == j)) for j in range(self.num_classes)] for i in range(self.num_classes)])


    def train(self, data: np.ndarray | ArrayLike, y: np.ndarray | ArrayLike):
        """
        Every child should implement this method. Keep this blank.

        :param data:
        :param y:
        """
        pass

    def predict(self, data: np.ndarray | ArrayLike, k: int):
        """
        Every child should implement this method. Keep this blank.

        :param data: ndarray. shape=(num_test_samps, num_features).
            Data to predict the class of.
        :param k: int.
            k for kNN.
        """
        pass
