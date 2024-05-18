"""data_transformations.py
YOUR NAME HERE
Performs translation, scaling, and rotation transformations on data
CS 251 / 252: Data Analysis and Visualization
Spring 2024

NOTE: All functions should be implemented from scratch using basic NumPy WITHOUT loops and high-level library calls.
"""
import numpy as np
import warnings as warn
from typing import Union


def normalize(data: np.ndarray) -> np.ndarray:
    """Perform min-max normalization of each variable in a dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be normalized.

    Returns:
    -----------
    ndarray. shape=(N, M). The min-max normalized dataset.
    """
    return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))


def center(data: np.ndarray) -> np.ndarray:
    """Center the dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be centered.

    Returns:
    -----------
    ndarray. shape=(N, M). The centered dataset.
    """
    return data - data.mean(axis=0)


def rotation_matrix_3d(degrees: Union[int, float, complex, np.number], axis='x') -> np.ndarray:
    """Make a 3D rotation matrix for rotating the dataset about ONE variable ("axis").

    Parameters:
    -----------
    degrees: float. Angle (in degrees) by which the dataset should be rotated.
    axis: str. Specifies the variable about which the dataset should be rotated. Assumed to be either 'x', 'y', or 'z'.

    Returns:
    -----------
    ndarray. shape=(3, 3). The 3D rotation matrix.

    NOTE: This method just CREATES and RETURNS the rotation matrix. It does NOT actually PERFORM the rotation!
    """
    if isinstance(degrees, complex) or isinstance(degrees, np.complex_):
        warn.warn("Complex numbers are not a degree measurement. "
                  "Converting to degrees by complex argument.", RuntimeWarning)
        degrees = np.angle(degrees, deg=True)
    radians = np.radians(degrees)
    if axis == 'x':
        outArray = np.array([[1, 0, 0],
                             [0, np.cos(radians), -np.sin(radians)],
                             [0, np.sin(radians), np.cos(radians)]])
    elif axis == 'y':
        outArray = np.array([[np.cos(radians), 0, np.sin(radians)],
                             [0, 1, 0],
                             [-np.sin(radians), 0, np.cos(radians)]])
    elif axis == 'z':
        outArray = np.array([[np.cos(radians), -np.sin(radians), 0],
                             [np.sin(radians), np.cos(radians), 0],
                             [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")
    return outArray


def rotate_data(data: np.ndarray, degrees: Union[int, float, complex, np.number], axis='x') -> np.ndarray:
    """Rotate the dataset about ONE variable ("axis").

    Parameters:
    -----------
    data: ndarray. shape=(N, 3). The dataset to be rotated.
    degrees: float. Angle (in degrees) by which the dataset should be rotated.
    axis: str. Specifies the variable about which the dataset should be rotated. Assumed to be either 'x', 'y', or 'z'.

    Returns:
    -----------
    ndarray. shape=(N, 3). The rotated dataset.
    """
    return data @ rotation_matrix_3d(degrees, axis)
