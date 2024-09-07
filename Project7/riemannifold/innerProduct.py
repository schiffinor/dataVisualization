import numpy as np
import numpyProj as npP
from typing import Tuple, List, Union, Dict, NewType, Callable
from Operands import *
import warnings as wn

# Define a type for norm functions
Norm = NewType("Norm", Callable[[np.ndarray, np.ndarray, ...], float])

Norm_ndim = NewType("Norm_ndim", Callable[[np.ndarray, np.ndarray, ...], np.ndarray])

Norm_plist = NewType("Norm_plist", Callable[[np.ndarray, np.ndarray, ...], np.ndarray])

# Define a type for inner product functions
InnerProd = NewType("InnerProduct", Callable[[np.ndarray, np.ndarray, ...], float])

InnerProd_ndim = NewType("InnerProduct_ndim", Callable[[np.ndarray, np.ndarray, ...], np.ndarray])

InnerProd_plist = NewType("InnerProduct_plist", Callable[[np.ndarray, np.ndarray, ...], np.ndarray])

# Define a type for coordinate functions
CoordTransform = NewType("CoordTransform", Callable[[np.ndarray], np.ndarray])


class InnerProduct:
    def __init__(self, inner_product: InnerProd, inner_product_ndim: InnerProd_ndim = None,
                 inner_product_plist: InnerProd_plist = None):
        self.inner_product_func = inner_product
        self.ndim_inner_product_func = inner_product_ndim
        self.plist_inner_product_func = inner_product_plist
        self.inner_product_args = []
        self.inner_product_kwargs = {}
        self.ndim_args = []
        self.ndim_kwargs = {}
        self.plist_args = []
        self.plist_kwargs = {}

    def fullSet(self, inner_product: InnerProd, inner_product_ndim: InnerProd_ndim = None,
                inner_product_plist: InnerProd_plist = None):
        self.inner_product_func = inner_product
        if inner_product_ndim is not None:
            self.ndim_inner_product_func = inner_product_ndim
        else:
            wn.warn(
                "No n-dimensional inner product function provided. "
                "Defaulting to For loop based implementation. "
                "Very slow for large datasets.")
            self.ndim_inner_product_func = self.forLoopNdim
            self.ndim_args = [inner_product]
        if inner_product_plist is not None:
            self.plist_inner_product_func = inner_product_plist
        else:
            wn.warn(
                "No parallel inner product function provided. "
                "Defaulting to For loop based implementation. "
                "Very slow for large datasets.")
            self.plist_inner_product_func = self.forLoopPlist
            self.plist_args = [inner_product]

    def forLoopNdim(self, a: np.ndarray, b: np.ndarray, inner_product: InnerProd = None) -> np.ndarray:
        """
        Calculate the inner product of two tensors of vectors using a for loop.
        """
        if inner_product is None:
            ValueError("No inner product function provided.")
        if a.shape != b.shape:
            raise ValueError("Vectors must have the same shape.")
        if a.ndim != b.ndim:
            raise ValueError("Vectors must have the same number of dimensions.")
        if a.ndim < 2:
            raise ValueError("Vectors must have at least 2 dimensions.")
        loops = a.ndim
        result = np.zeros(a.shape[:-1])
        if loops == 2:
            for i in range(a.shape[0]):
                result[i] = inner_product(a[i], b[i])
        else:
            for i in range(a.shape[0]):
                result[i] = self.forLoopNdim(a[i], b[i], inner_product)
        return result

    def forLoopPlist(self, a: np.ndarray, b: np.ndarray, inner_product: InnerProd = None) -> np.ndarray:
        """
        Calculate the inner product of two tensors of vectors using a for loop.
        """
        if inner_product is None:
            ValueError("No inner product function provided.")
        if a.shape != b.shape:
            raise ValueError("Vectors must have the same shape.")
        if a.ndim != b.ndim:
            raise ValueError("Vectors must have the same number of dimensions.")
        if a.ndim != 2:
            raise ValueError("Arrays must have 2 dimensions. Must be a list of vectors.")
        result = np.zeros([a.shape[0], b.shape[0]])
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                result[i, j] = inner_product(a[i], b[j])
        return result

    def getInnerProduct(self):
        return self.inner_product_func

    def getNdimInnerProduct(self):
        return self.ndim_inner_product_func

    def getInnerProductPlist(self):
        return self.plist_inner_product_func

    def getInnerProductArgs(self):
        return self.inner_product_args

    def getInnerProductKwargs(self):
        return self.inner_product_kwargs

    def getNdimArgs(self):
        return self.ndim_args

    def getNdimKwargs(self):
        return self.ndim_kwargs

    def getPlistArgs(self):
        return self.plist_args

    def setInnerProduct(self, inner_product: InnerProd):
        self.inner_product_func = inner_product

    def setNdimInnerProduct(self, inner_product_ndim: InnerProd_ndim):
        self.ndim_inner_product_func = inner_product_ndim

    def setInnerProductPlist(self, inner_product_plist: InnerProd_plist):
        self.plist_inner_product_func = inner_product_plist

    def setInnerProductArgs(self, *args):
        self.inner_product_args = args

    def setInnerProductKwargs(self, **kwargs):
        self.inner_product_kwargs = kwargs

    def setNdimArgs(self, *args):
        self.ndim_args = args

    def setNdimKwargs(self, **kwargs):
        self.ndim_kwargs = kwargs

    def setPlistArgs(self, *args):
        self.plist_args = args

    def setPlistKwargs(self, **kwargs):
        self.plist_kwargs = kwargs

    def inProd(self, a: np.ndarray, b: np.ndarray) -> float:
        return self.inner_product_func(a, b, *self.inner_product_args, **self.inner_product_kwargs)

    def inProdNdim(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.ndim_inner_product_func(a, b, *self.ndim_args, **self.ndim_kwargs)

    def inProdPlist(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.plist_inner_product_func(a, b, *self.plist_args, **self.plist_kwargs)

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        return self.inProd(a, b)

    def __str__(self):
        return f"InnerProduct object with function {self.inner_product_func.__name__}."

    def __repr__(self):
        return f"InnerProduct object with function {self.inner_product_func.__name__}."

    def __eq__(self, other):
        random_a = np.random.rand(4, 1)
        random_b = np.random.rand(4, 1)
        return self.inProd(random_a, random_b) == other.inProd(random_a, random_b)

    def __gt__(self, other):
        random_a = np.random.rand(10, 1)
        random_b = np.random.rand(10, 1)
        return self.inProd(random_a, random_b) > other.inProd(random_a, random_b)

    def __lt__(self, other):
        random_a = np.random.rand(10, 1)
        random_b = np.random.rand(10, 1)
        return self.inProd(random_a, random_b) < other.inProd(random_a, random_b)

    def __ge__(self, other):
        random_a = np.random.rand(10, 1)
        random_b = np.random.rand(10, 1)
        return self.inProd(random_a, random_b) >= other.inProd(random_a, random_b)

    def __le__(self, other):
        random_a = np.random.rand(10, 1)
        random_b = np.random.rand(10, 1)
        return self.inProd(random_a, random_b) <= other.inProd(random_a, random_b)

    def __ne__(self, other):
        random_a = np.random.rand(4, 1)
        random_b = np.random.rand(4, 1)
        return self.inProd(random_a, random_b) != other.inProd(random_a, random_b)

    def __add__(self, other):
        if isinstance(other, InnerProduct):
            return InnerProduct(lambda x, y: self.inProd(x, y) + other.inProd(x, y),
                                lambda x, y: self.inProdNdim(x, y) + other.inProdNdim(x, y),
                                lambda x, y: self.inProdPlist(x, y) + other.inProdPlist(x, y))
        else:
            return InnerProduct(lambda x, y: self.inProd(x, y) + other,
                                lambda x, y: self.inProdNdim(x, y) + other,
                                lambda x, y: self.inProdPlist(x, y) + other)

    def __sub__(self, other):
        if isinstance(other, InnerProduct):
            return InnerProduct(lambda x, y: self.inProd(x, y) - other.inProd(x, y),
                                lambda x, y: self.inProdNdim(x, y) - other.inProdNdim(x, y),
                                lambda x, y: self.inProdPlist(x, y) - other.inProdPlist(x, y))
        else:
            return InnerProduct(lambda x, y: self.inProd(x, y) - other,
                                lambda x, y: self.inProdNdim(x, y) - other,
                                lambda x, y: self.inProdPlist(x, y) - other)

    def __mul__(self, other):
        if isinstance(other, InnerProduct):
            return InnerProduct(lambda x, y: self.inProd(x, y) * other.inProd(x, y),
                                lambda x, y: self.inProdNdim(x, y) * other.inProdNdim(x, y),
                                lambda x, y: self.inProdPlist(x, y) * other.inProdPlist(x, y))
        else:
            return InnerProduct(lambda x, y: self.inProd(x, y) * other,
                                lambda x, y: self.inProdNdim(x, y) * other,
                                lambda x, y: self.inProdPlist(x, y) * other)

    def __truediv__(self, other):
        if isinstance(other, InnerProduct):
            return InnerProduct(lambda x, y: self.inProd(x, y) / other.inProd(x, y),
                                lambda x, y: self.inProdNdim(x, y) / other.inProdNdim(x, y),
                                lambda x, y: self.inProdPlist(x, y) / other.inProdPlist(x, y))
        else:
            return InnerProduct(lambda x, y: self.inProd(x, y) / other,
                                lambda x, y: self.inProdNdim(x, y) / other,
                                lambda x, y: self.inProdPlist(x, y) / other)

    def __floordiv__(self, other):
        if isinstance(other, InnerProduct):
            return InnerProduct(lambda x, y: self.inProd(x, y) // other.inProd(x, y),
                                lambda x, y: self.inProdNdim(x, y) // other.inProdNdim(x, y),
                                lambda x, y: self.inProdPlist(x, y) // other.inProdPlist(x, y))
        else:
            return InnerProduct(lambda x, y: self.inProd(x, y) // other,
                                lambda x, y: self.inProdNdim(x, y) // other,
                                lambda x, y: self.inProdPlist(x, y) // other)

    def __mod__(self, other):
        if isinstance(other, InnerProduct):
            return InnerProduct(lambda x, y: self.inProd(x, y) % other.inProd(x, y),
                                lambda x, y: self.inProdNdim(x, y) % other.inProdNdim(x, y),
                                lambda x, y: self.inProdPlist(x, y) % other.inProdPlist(x, y))
        else:
            return InnerProduct(lambda x, y: self.inProd(x, y) % other,
                                lambda x, y: self.inProdNdim(x, y) % other,
                                lambda x, y: self.inProdPlist(x, y) % other)

    def __pow__(self, other):
        if isinstance(other, InnerProduct):
            return InnerProduct(lambda x, y: self.inProd(x, y) ** other.inProd(x, y),
                                lambda x, y: self.inProdNdim(x, y) ** other.inProdNdim(x, y),
                                lambda x, y: self.inProdPlist(x, y) ** other.inProdPlist(x, y))
        else:
            return InnerProduct(lambda x, y: self.inProd(x, y) ** other,
                                lambda x, y: self.inProdNdim(x, y) ** other,
                                lambda x, y: self.inProdPlist(x, y) ** other)
