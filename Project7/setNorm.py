import numpy as np
from typing import Tuple, List, Union, Dict, NewType, Callable

import analysis
import kmeans
import analysis as an
import warnings as wn

# Define a type for a function that takes two np.ndarrays and returns a scalar
Norm = NewType("Norm", Callable[[np.ndarray, np.ndarray, ...], float])

Norm_ndim = NewType("Norm_ndim", Callable[[np.ndarray, np.ndarray, ...], np.ndarray])

Norm_plist = NewType("Norm_plist", Callable[[np.ndarray, np.ndarray, ...], np.ndarray])


class setNorm:
    def __init__(self, norm: Norm, norm_ndim: Norm_ndim = None, norm_plist: Norm_plist = None,
                 normP: int | float = None):
        self.norm_func = norm
        self.ndim_norm_func = None
        self.plist_norm_func = None
        self.normP = None
        self.custom = False
        self.norm_args = []
        self.norm_kwargs = {}
        self.ndim_args = []
        self.ndim_kwargs = {}
        self.plist_args = []
        self.plist_kwargs = {}



    def getNorm(self):
        return self.norm_func

    def getNormP(self):
        return self.normP

    def getNormNdim(self):
        return self.ndim_norm_func

    def getNormPlist(self):
        return self.plist_norm_func

    def getNormArgs(self):
        return self.norm_args

    def getNormKwargs(self):
        return self.norm_kwargs

    def getNdimArgs(self):
        return self.ndim_args

    def getNdimKwargs(self):
        return self.ndim_kwargs

    def getPlistArgs(self):
        return self.plist_args

    def getPlistKwargs(self):
        return self.plist_kwargs

    def getCustom(self):
        return self.custom

    def setNorm(self, norm: Norm):
        self.norm_func = norm

    def setNormP(self, normP: int | float):
        self.normP = normP

    def setNormNdim(self, norm_ndim: Norm_ndim):
        self.ndim_norm_func = norm_ndim

    def setNormPlist(self, norm_plist: Norm_plist):
        self.plist_norm_func = norm_plist

    def setNormArgs(self, norm_args: List):
        self.norm_args = norm_args

    def setNormKwargs(self, norm_kwargs: Dict):
        self.norm_kwargs = norm_kwargs

    def setNdimArgs(self, ndim_args: List):
        self.ndim_args = ndim_args

    def setNdimKwargs(self, ndim_kwargs: Dict):
        self.ndim_kwargs = ndim_kwargs

    def setPlistArgs(self, plist_args: List):
        self.plist_args = plist_args

    def setPlistKwargs(self, plist_kwargs: Dict):
        self.plist_kwargs = plist_kwargs

    def setCustom(self, custom: bool):
        self.custom = custom

    def setNormAll(self, norm: Norm, normP: int | float, norm_ndim: Norm_ndim, norm_plist: Norm_plist):
        pVal = None
        try:
            pVal = kmeans.KMeans.norm_validation(norm, normP)
        except:
            self.custom = True
            wn.warn("Norm not in predefined set of norms. Make sure to provide functional ndim and plist norms.")
            if norm_ndim is None:
                raise ValueError("norm_ndim must be provided if norm is not in predefined set of norms.")
            if norm_plist is None:
                raise ValueError("norm_plist must be provided if norm is not in predefined set of norms.")

        if self.custom is False:
            self.ndim_norm_func = kmeans.KMeans.ndim_norm_choice(norm)
            self.plist_norm_func = an.Analysis.lp_norm_v2_pList
            acc_norm_names = ["0", "1", "2", "i"]
            acc_norm_nums = [0, 1, 2, np.inf]
            norms = [analysis.Analysis.l0_norm, analysis.Analysis.l1_norm,
                     analysis.Analysis.l2_norm, analysis.Analysis.l_inf_norm]
            norm_num_dic = dict(zip(acc_norm_nums, norms))
            norm_name_dic = dict(zip(acc_norm_names, norms))
            norm_name_num_dic = dict(zip(acc_norm_names, acc_norm_nums))
            self.norm_args = []
            self.norm_kwargs = {}
            self.ndim_args = []
            self.ndim_kwargs = {}
            self.plist_args = []
            if pVal != "p":
                self.norm_func = norm_name_dic[pVal]
                self.normP = norm_name_num_dic[pVal]
                self.ndim_norm_func = kmeans.KMeans.ndim_norm_choice(self.norm_func)
            elif normP in acc_norm_nums:
                self.norm_func = norm_num_dic[normP]
                self.normP = normP
                self.ndim_norm_func = kmeans.KMeans.ndim_norm_choice(self.norm_func)
            else:
                self.normP = normP
                self.norm_kwargs = {"p": self.normP}
                self.ndim_kwargs = {"p": self.normP}
            self.norm_args = []
            self.plist_kwargs = {"p": self.normP, "debug": False}
        else:
            self.normP = normP
            self.ndim_norm_func = norm_ndim
            self.plist_norm_func = norm_plist

    def getNormAll(self):
        return self.norm_func, self.normP, self.ndim_norm_func, self.plist_norm_func

    def getNormArgsAll(self):
        return self.norm_args, self.norm_kwargs, self.ndim_args, self.ndim_kwargs, self.plist_args, self.plist_kwargs

    def pt_dist(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute the distance between two points

        Parameters:
        -----------
        x: ndarray. shape=(num_features,). Point x
        y: ndarray. shape=(num_features,). Point y

        Returns:
        -----------
        float. Distance between x and y
        """
        return self.norm_func(x, y, *self.norm_args, **self.norm_kwargs)

    def ndim_dist(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute the distance between two n-dimensional points

        Parameters:
        -----------
        x: ndarray. shape=(num_samps, num_features). Points x
        y: ndarray. shape=(num_samps, num_features). Points y

        Returns:
        -----------
        ndarray. shape=(num_samps,). Distance between x and y
        """
        return self.ndim_norm_func(x, y, *self.ndim_args, **self.ndim_kwargs)

    def plist_dist(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute the distance between two lists of n-dimensional points

        Parameters:
        -----------
        x: List[ndarray]. List of points x
        y: List[ndarray]. List of points y

        Returns:
        -----------
        ndarray. shape=(num_samps, num_samps). Distance between x and y
        """
        return self.plist_norm_func(x, y, *self.plist_args, **self.plist_kwargs)

    def norm(self, x: np.ndarray) -> float:
        return self.norm_func(x, np.zeros_like(x), *self.norm_args, **self.norm_kwargs)

    def ndim_norm(self, x: np.ndarray) -> np.ndarray:
        return self.ndim_norm_func(x, np.zeros_like(x), *self.ndim_args, **self.ndim_kwargs)

    def plist_norm(self, x: np.ndarray) -> np.ndarray:
        return self.plist_norm_func(x, np.zeros([1, x.shape[1]]), *self.plist_args, **self.plist_kwargs)

    def __str__(self):
        return f"Norm: {self.norm_func}, NormP: {self.normP}, Norm_ndim: {self.ndim_norm_func}, Norm_plist: {self.plist_norm_func}, Custom: {self.custom}"

    def __repr__(self):
        return f"Norm: {self.norm_func}, NormP: {self.normP}, Norm_ndim: {self.ndim_norm_func}, Norm_plist: {self.plist_norm_func}, Custom: {self.custom}"

    def __eq__(self, other):
        if not isinstance(other, setNorm):
            return False
        return self.norm_func == other.norm_func and self.normP == other.normP and self.ndim_norm_func == other.ndim_norm_func and self.plist_norm_func == other.plist_norm_func and self.custom == other.custom

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.norm_func, self.normP, self.ndim_norm_func, self.plist_norm_func, self.custom))

    def __copy__(self):
        return setNorm(self.norm_func, self.ndim_norm_func, self.plist_norm_func, self.normP)

    def __deepcopy__(self, memo):
        return setNorm(self.norm_func, self.ndim_norm_func, self.plist_norm_func, self.normP)

    def __getstate__(self):
        return self.norm_func, self.normP, self.ndim_norm_func, self.plist_norm_func, self.custom

    def __setstate__(self, state):
        self.norm_func, self.normP, self.ndim_norm_func, self.plist_norm_func, self.custom = state

    def __reduce__(self):
        return setNorm, (self.norm_func, self.ndim_norm_func, self.plist_norm_func, self.normP)

    def __reduce_ex__(self, protocol):
        return setNorm, (self.norm_func, self.ndim_norm_func, self.plist_norm_func, self.normP), (self.custom,)