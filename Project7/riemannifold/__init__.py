import numpy as np
import scipy as scp
import numpyProj as npP
from typing import Tuple, List, Union, Dict, NewType, Callable, Any

# Import custom modules and external libraries for specialized functions
from Operands import *
import warnings as wn
import innerProduct as iP
import setNorm as sN
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array
from sympy.tensor.array.expressions.from_array_to_matrix import convert_array_to_matrix
import sympy.tensor as st
import sympy.vector as sv
import sympy.matrices as sm
from sympy import init_printing
import sympy.diffgeom.diffgeom as dg

# Define a type for a function that takes a list of coordinates and returns a np.ndarray of coordinates,
# a parameterized function.
paramFN = NewType("paramFN", Callable[[Union[List[Union[int, float]], np.ndarray]], np.ndarray | sp.Array | sv.Vector])


# Function that applies a matrix function to arrays by converting them to matrices first
def feedThroughArrMat(arrays: sp.Array | List[sp.Array],
                      func: Callable[[Union[sp.Matrix, List[sp.Matrix]]], sp.Matrix] | Callable[
                          [Union[sp.Matrix, List[sp.Matrix]], ...], sp.Matrix],
                      batch: bool = False, *args, **kwargs) -> Union[sp.Array, List[sp.Array]]:
    """
    This function takes an array or list of arrays if necessary and a function that takes a sympy matrix and returns a sympy matrix.
    It transforms the arrays into matrices and applies the function to the matrix / matrices.
    It then transforms the matrices back into arrays and returns the array / arrays.
    :param arrays:
    :param func:
    :param batch:
    :param args:
    :param kwargs:
    :return:
    """
    if isinstance(arrays, list):
        if any(not isinstance(a, sp.Array) for a in arrays):
            raise ValueError("Arrays must be sympy arrays.")
        if batch:
            return [sp.Array(func(a.tomatrix(), *args, **kwargs).tolist()) for a in arrays]
        else:
            return sp.Array(func([a.tomatrix() for a in arrays], *args, **kwargs).tolist())
    elif isinstance(arrays, sp.Array):
        if batch:
            raise ValueError("If batch is True, arrays must be a list of sympy arrays.")
        else:
            return sp.Array(func(arrays.tomatrix(), *args, **kwargs).tolist())
    else:
        raise ValueError("Arrays must be a sympy array or a list of sympy arrays.")


def feedThroughMatArr(matrices: sp.Matrix | List[sp.Matrix],
                      func: Callable[[Union[sp.Array, List[sp.Array]]], sp.Array]
                            | Callable[[Union[sp.Array, List[sp.Array]], ...], sp.Array],
                      batch: bool = False, *args, **kwargs) -> Union[sp.Matrix, List[sp.Matrix]]:
    """
    This function takes a matrix or list of matrices if necessary and a function that takes a sympy array and returns a sympy array.
    It transforms the matrices into arrays and applies the function to the array / arrays.
    It then transforms the arrays back into matrices and returns the matrix / matrices.

    :param matrices:
    :param func:
    :param batch:
    :param args:
    :param kwargs:
    :return:
    """
    if isinstance(matrices, list):
        if any(not isinstance(m, sp.Matrix) for m in matrices):
            raise ValueError("Matrices must be sympy matrices.")
        if batch:
            return [func(sp.Array(m.tolist()), *args, **kwargs).tomatrix() for m in matrices]
        else:
            return (func([sp.Array(m.tolist()) for m in matrices], *args, **kwargs)).tomatrix()
    elif isinstance(matrices, sp.Matrix):
        if batch:
            raise ValueError("If batch is True, matrices must be a list of sympy matrices.")
        else:
            return (func(sp.Array(matrices.tolist()), *args, **kwargs)).tomatrix()
    else:
        raise ValueError("Matrices must be a sympy matrix or a list of sympy matrices.")


def paramToFn(variables: List[str], param: List[str],
              checker: bool = False, simplify: bool = False, latex: bool = False) -> paramFN:
    """
    This function takes a list of variables and a list of parameters and returns a function that takes a list of
    coordinates and returns a np.ndarray of coordinates.
    Variable names should be one word, and parameters should be valid sympy expressions.
    Parameters are parsed using sympy.parsing.sympy_parser.parse_expr.
    Parametrization will be printed in the form as a list of sympy expressions.
    If the parametrization is invalid, the function will raise a SyntaxError.
    If the parsed parametrization does not match intended use, rerun the function with corrected parameters.
    """
    if variables is None or param is None:
        raise ValueError("Variables and parameters must be lists of strings.")
    if any(" " in v for v in variables):
        raise ValueError("Variable names should be one word.")
    if any("i" == v for v in variables):
        raise ValueError("Variable names should not be 'i'.")

    # Goes through list of variables and creates a list of sympy symbols
    var = [sp.symbols(v) for v in variables]

    # Creates local dictionary of variables
    varDic = dict(zip(variables, var))

    # Goes through parametrization and creates a list of sympy expressions
    if latex:
        parametric = [parse_latex(r'{}'.format(p), local_dict=varDic, backend="Lark") for p in param]
    else:
        parametric = [parse_expr(p, local_dict=varDic, transformations="all") for p in param]

    # Prints the parametrization if checker is True
    if checker:
        parametricStr = [str(p) for p in parametric]
        latexParametric = [sp.latex(p) for p in parametric]
        print(f"Parametrization: {parametricStr}")
        print(f"Latex Parametrization: {latexParametric}")
        # Pretty Print
        init_printing()
        for p in parametric:
            sp.pprint(p, use_unicode=True)

    # Simplifies the parametrization if simplify is True
    if simplify:
        parametric = [sp.simplify(p) for p in parametric]
        if checker:
            parametricStr = [str(p) for p in parametric]
            latexParametric = [sp.latex(p) for p in parametric]
            print(f"Simplified Parametrization: {parametricStr}")
            print(f"Simplified Latex Parametrization: {latexParametric}")
            # Pretty Print
            init_printing()
            for p in parametric:
                sp.pprint(p, use_unicode=True)

    # Create a function that takes a list of coordinates and returns a np.ndarray of coordinates
    lamFN = sp.lambdify(var, parametric, modules="numpy", cse=True)

    def parametricFN(coordinates: List[Union[int, float]] | np.ndarray) -> np.ndarray:
        return np.array(lamFN(*(coordinates if isinstance(coordinates, list) else coordinates.tolist())))

    return parametricFN


def batchParamToFn(variables: List[str] = None, param: List[str] = None, parameterization: paramFN = None,
                   checker: bool = False, simplify: bool = False, latex: bool = False) -> paramFN:
    """
    This function takes a list of variables and a list of parameters and returns a function that takes a list of
    coordinates and returns a np.ndarray of coordinates.
    Variable names should be one word, and parameters should be valid sympy expressions.
    Parameters are parsed using sympy.parsing.sympy_parser.parse_expr.
    Parametrization will be printed in the form as a list of sympy expressions.
    If the parametrization is invalid, the function will raise a SyntaxError.
    If the parsed parametrization does not match intended use, rerun the function with corrected parameters.
    """
    if parameterization is None:
        if variables is None or param is None:
            raise ValueError("Variables and parameters must be lists of strings.")
        paramFunc = paramToFn(variables, param, checker=checker, simplify=simplify, latex=latex)
    else:
        paramFunc = parameterization

    def batchParametricFN(coordinates: List[List[Union[int, float]]] | List[np.ndarray] | np.ndarray) -> np.ndarray:
        if isinstance(coordinates, list):
            if isinstance(coordinates[0], list) or isinstance(coordinates[0], np.ndarray):
                return np.array([paramFunc(c) for c in coordinates])
            else:
                raise ValueError("Coordinates must be a list of lists or a list of np.ndarrays. OR alternatively, "
                                 "a np.ndarray.")
        elif isinstance(coordinates, np.ndarray):
            if len(coordinates.shape) == 2:
                return np.array([paramFunc(c) for c in coordinates])
            else:
                raise ValueError("Coordinate list as an np.ndarray must be 2D.")

    return batchParametricFN


def paramToSympy(variables: List[str], param: List[str],
                 checker: bool = False, simplify: bool = False, latex: bool = False) -> paramFN:
    """
    This function takes a list of variables and a list of parameters and returns a function that takes a list of
    coordinates and returns a np.ndarray of coordinates.
    Variable names should be one word, and parameters should be valid sympy expressions.
    Parameters are parsed using sympy.parsing.sympy_parser.parse_expr.
    Parametrization will be printed in the form as a list of sympy expressions.
    If the parametrization is invalid, the function will raise a SyntaxError.
    If the parsed parametrization does not match intended use, rerun the function with corrected parameters.
    """
    if variables is None or param is None:
        raise ValueError("Variables and parameters must be lists of strings.")
    if any(" " in v for v in variables):
        raise ValueError("Variable names should be one word.")
    if any("i" == v for v in variables):
        raise ValueError("Variable names should not be 'i'.")

    # Goes through list of variables and creates a list of sympy symbols
    var = [sp.symbols(v) for v in variables]

    # Creates local dictionary of variables
    varDict = dict(zip(variables, var))

    # Goes through parametrization and creates a list of sympy expressions
    if latex:
        parametric = [parse_latex(r'{}'.format(p), local_dict=varDict, backend="Lark") for p in param]
    else:
        parametric = [parse_expr(p, local_dict=varDict, transformations="all") for p in param]

    # Prints the parametrization if checker is True
    if checker:
        parametricStr = [str(p) for p in parametric]
        latexParametric = [sp.latex(p) for p in parametric]
        print(f"Parametrization: {parametricStr}")
        print(f"Latex Parametrization: {latexParametric}")
        # Pretty Print
        init_printing()
        for p in parametric:
            sp.pprint(p, use_unicode=True)

    # Simplifies the parametrization if simplify is True
    if simplify:
        parametric = [sp.simplify(p) for p in parametric]
        if checker:
            parametricStr = [str(p) for p in parametric]
            latexParametric = [sp.latex(p) for p in parametric]
            print(f"Simplified Parametrization: {parametricStr}")
            print(f"Simplified Latex Parametrization: {latexParametric}")
            # Pretty Print
            init_printing()
            for p in parametric:
                sp.pprint(p, use_unicode=True)

    return parametric


def paramToSympyArray(parametrization: List[sp.Expr]) -> sp.symarray:
    """
    This function takes a list of sympy expressions and returns a sympy symarray.
    """
    out = sp.symarray("g", len(parametrization))
    for i, p in enumerate(parametrization):
        out[i] = p
    return out


class varDict:
    """
    This class is a dictionary of variables.
    """

    def __init__(self, var: List[str]) -> None:
        self.variables = var
        self.varDic = dict(zip(var, [sp.symbols(v) for v in var]))
        for var in var:
            setattr(self, var, self.varDic[var])

    def getKeys(self) -> List[str]:
        return list(self.varDic.keys())

    def getValues(self) -> List[sp.Symbol]:
        return list(self.varDic.values())

    def getItems(self) -> List[Tuple[str, sp.Symbol]]:
        return list(self.varDic.items())

    def getDict(self) -> Dict[str, sp.Symbol]:
        return self.varDic

    def getVar(self, key: str) -> sp.Symbol:
        return self.varDic[key]

    def __getitem__(self, key: str) -> sp.Symbol:
        return self.varDic[key]

    def __setitem__(self, key: str, value: sp.Symbol) -> None:
        self.varDic[key] = value
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        del self.varDic[key]
        delattr(self, key)

    def __len__(self) -> int:
        return len(self.varDic)

    def __iter__(self) -> iter:
        return iter(self.varDic)

    def __str__(self) -> str:
        return str(self.varDic)

    def __repr__(self) -> str:
        return repr(self.varDic)

    def __contains__(self, key: str) -> bool:
        return key in self.varDic

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, varDict):
            return False
        return self.varDic == other.varDic

    def __ne__(self, other: Any) -> bool:
        if not isinstance(other, varDict):
            wn.warn("Comparison between varDict and non-varDict object.")
            return True
        return self.varDic != other.varDic

    def __add__(self, other: Any) -> Any:
        if not isinstance(other, varDict):
            if isinstance(other, list):
                wn.warn("Addition between varDict and non-varDict object.")
                return varDict(list(self.varDic.keys()) + other)
            elif isinstance(other, dict):
                wn.warn("Addition between varDict and non-varDict object.")
                return varDict(list(self.varDic.keys()) + list(other.keys()))
            else:
                raise ValueError("Addition must be between two varDict objects.")
        return varDict(list(self.varDic.keys()) + list(other.varDic.keys()))

    def __sub__(self, other: Any) -> Any:
        if not isinstance(other, varDict):
            if isinstance(other, list):
                wn.warn("Subtraction between varDict and non-varDict object.")
                return varDict([ct for ct in self.varDic.keys() if ct not in other])
            elif isinstance(other, dict):
                wn.warn("Subtraction between varDict and non-varDict object.")
                return varDict([ct for ct in self.varDic.keys() if ct not in other.keys()])
            else:
                raise ValueError("Subtraction must be between two varDict objects.")
        return varDict([ct for ct in self.varDic.keys() if ct not in other.varDic.keys()])

    def __mul__(self, other: Any) -> Any:
        if not isinstance(other, varDict):
            if isinstance(other, list):
                wn.warn("Multiplication between varDict and non-varDict object.")
                return varDict(list(set(self.varDic.keys()) & set(other)))
            elif isinstance(other, dict):
                wn.warn("Multiplication between varDict and non-varDict object.")
                return varDict(list(set(self.varDic.keys()) & set(other.keys())))
            else:
                raise ValueError("Multiplication must be between two varDict objects.")
        return varDict(list(set(self.varDic.keys()) & set(other.varDic.keys())))

    def __truediv__(self, other: Any) -> Any:
        if not isinstance(other, varDict):
            if isinstance(other, list):
                wn.warn("Division between varDict and non-varDict object.")
                return varDict(list(set(self.varDic.keys()) ^ set(other)))
            elif isinstance(other, dict):
                wn.warn("Division between varDict and non-varDict object.")
                return varDict(list(set(self.varDic.keys()) ^ set(other.keys())))
            else:
                raise ValueError("Division must be between two varDict objects.")
        return varDict(list(set(self.varDic.keys()) ^ set(other.varDic.keys())))


class coordDict:
    """
    This class is a dictionary of coordinate systems.
    """

    def __init__(self, coordPatch: dg.Patch, maniName: str, name: str = None) -> None:
        self.patch = coordPatch
        self.mName = maniName
        self.name = name if name is not None else "CS"
        self.cSDict = {}
        self.cSList = []
        self.relDict = {}

    def addCoordSysPRE(self, cS: dg.CoordSystem) -> None:
        self.cSDict[cS.name] = cS
        self.cSList.append(cS)
        if cS.name in self.__dict__:
            raise ValueError(f"coordDict has parameter {cS.name} already. Banned parameter names are: {self.__dict__}")
        setattr(self, str(cS.name), cS)

    def addCoordSys(self, fromName: str, toName: str, fromVars: List[sp.Symbol], transform: List[sp.Expr]) -> None:
        if not isinstance(fromVars, list):
            raise ValueError("fromVars must be a list of sympy symbols.")
        if not isinstance(transform, list):
            raise ValueError("transform must be a list of sympy expressions.")
        if not isinstance(fromName, str):
            raise ValueError("fromName must be a string.")
        if not isinstance(toName, str):
            raise ValueError("toName must be a string.")
        self.relDict[(fromName, toName)] = [tuple(fromVars), tuple(transform)]
        cS = dg.CoordSystem(fromName, self.patch, fromVars, relations=self.relDict)
        self.addCoordSysPRE(cS)

    def getKeys(self) -> List[str]:
        return list(self.cSDict.keys())

    def getValues(self) -> List[dg.CoordSystem]:
        return list(self.cSDict.values())

    def getItems(self) -> List[Tuple[str, dg.CoordSystem]]:
        return list(self.cSDict.items())

    def getDict(self) -> Dict[str, dg.CoordSystem]:
        return self.cSDict

    def getCoordSys(self, key: str) -> dg.CoordSystem:
        return self.cSDict[key]

    def getRD(self) -> Dict[Tuple[str, str], List[Union[List[sp.Symbol], List[sp.Expr]]]]:
        return self.relDict

    def getRDKeys(self) -> List[Tuple[str, str]]:
        return list(self.relDict.keys())

    def getRDValues(self) -> List[List[Union[List[sp.Symbol], List[sp.Expr]]]]:
        return list(self.relDict.values())

    def __getitem__(self, key: str) -> dg.CoordSystem:
        return self.cSDict[key]

    def __setitem__(self, key: str, value: dg.CoordSystem) -> None:
        self.cSDict[key] = value
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        del self.cSDict[key]
        delattr(self, key)

    def __len__(self) -> int:
        return len(self.cSDict)

    def __iter__(self) -> iter:
        return iter(self.cSDict)

    def __str__(self) -> str:
        return str(self.cSDict)

    def __repr__(self) -> str:
        return repr(self.cSDict)

    def __contains__(self, key: str) -> bool:
        return key in self.cSDict


class RiemannianManifold:

    def __init__(self, dim: int, variables: List[str], parameterization: List[str],
                 boundaries: List[Tuple[float]] = None, verbose: bool = False, latex: bool = False,
                 name: str = None, pName: str = None, cName: str = None, embeddingDim: int = None,
                 embeddingVars: List[str] = None, inverseParameterization: List[str] = None) -> None:
        if dim < 1:
            raise ValueError("Dimension must be greater than 0.")
        self.dim = dim
        if len(variables) != dim:
            raise ValueError("Number of variables must match the dimension.")
        if embeddingDim is not None:
            if embeddingDim < dim:
                raise ValueError("Embedding dimension must be greater than or equal to the dimension.")
            else:
                self.embeddingDim = embeddingDim
                if embeddingVars is not None:
                    if len(embeddingVars) != embeddingDim:
                        raise ValueError("Number of embedding variables must match the embedding dimension.")
                    self.embedding_variable_names = embeddingVars
                else:
                    self.embedding_variable_names = [f"e_{ct}" for ct in range(embeddingDim)]
        else:
            if embeddingVars is not None:
                if len(embeddingVars) < dim:
                    raise ValueError("Number of embedding variables must be greater than or equal to the dimension.")
                else:
                    self.embeddingDim = len(embeddingVars)
                    self.embedding_variable_names = [f"e_{ct}" for ct in range(self.embeddingDim)]
            else:
                self.embeddingDim = dim
                self.embedding_variable_names = [f"e_{ct}" for ct in range(self.embeddingDim)]
        self.manifold = dg.Manifold((name if name is not None else "M"), dim)
        self.patch = dg.Patch((pName if pName is not None else "P"), self.manifold)
        self.vars = varDict(variables)
        self.eVars = varDict(self.embedding_variable_names)
        self.variable_names = variables
        self.variables = self.vars.getValues()
        self.embeddingVars = self.eVars.getValues()
        # Remember to change embedding var Names
        self.strParam = parameterization
        self.strInvParam = inverseParameterization
        if isinstance(parameterization, list):
            print("Generating parametric function.")
            self.parametricFN = paramToFn(variables, parameterization, verbose, latex=latex, simplify=True)
        else:
            raise ValueError("Parameterization must be a list of strings or a parametric function.")
        self.batchParametricFN = batchParamToFn(parameterization=self.parametricFN)
        if inverseParameterization is not None:
            if embeddingVars is None:
                raise ValueError("Embedding variables must be provided for inverse parameterization.")
            if embeddingDim is None:
                raise ValueError("Embedding dimension must be provided for inverse parameterization.")
            if len(inverseParameterization) != dim:
                raise ValueError("Number of inverse parameters must match the dimension of the manifold.")
            if isinstance(inverseParameterization, list):
                print("Generating inverse parametric function.")
                self.inverseParametricFN = paramToFn(embeddingVars, inverseParameterization, verbose, latex=latex,
                                                     simplify=True)
            else:
                raise ValueError("Inverse parameterization must be a list of strings or a parametric function.")
            self.batchInverseParametricFN = batchParamToFn(parameterization=self.inverseParametricFN)
        if boundaries is not None:
            if len(boundaries) != dim:
                raise ValueError("Boundaries must have the same length as the dimension.")
            if any(len(b) != 2 for b in boundaries):
                raise ValueError("Each boundary must have two values.")
            if any(b[0] >= b[1] for b in boundaries):
                raise ValueError("The first value of each boundary must be less than the second value.")
            if len(variables) != len(boundaries):
                raise ValueError("Boundaries must have the same length as the variables.")
            self.boundaries = dict(zip(variables, boundaries))
        else:
            self.boundaries = None
        self.name = name if name is not None else "M"
        self.pName = pName
        self.verbose = verbose
        self.listParam = paramToSympy(variables, parameterization, latex=latex)
        self.matrixParam = sp.Matrix(self.listParam)
        self.arrayParam = sp.Array(self.listParam)
        self.coordSysts = coordDict(self.patch, self.name)
        self.coordSysts.addCoordSys(self.name, "Euclidean", self.variables, self.listParam)
        if inverseParameterization is not None:
            self.matrixInvParam = paramToSympy(embeddingVars, inverseParameterization, latex=latex).tomatrix()
            self.arrayInvParam = convert_matrix_to_array(self.matrixInvParam)
            self.listInvParam = self.matrixInvParam.tolist()
            self.coordSysts.addCoordSys("Euclidean", self.name, self.embeddingVars, self.listInvParam)
        cNameSet = cName if cName is not None else "CS"

        # Metric Tensor
        self.metricTensor = None
        # Inverse Metric Tensor
        self.inverseMetricTensor = None
        # Jacobian
        self.jacobian = None
        # Christoffel Symbols
        self.christoffels = None
        # Levi-Civita Connection
        self.leviCivita = None
        # Curvature Tensor
        self.curvatureTensor = None
        # Ricci Tensor
        self.ricciTensor = None
        # Ricci Scalar
        self.ricciScalar = None
        # Einstein Tensor
        self.einsteinTensor = None
        # Scalar Curvature
        self.scalarCurvature = None

    @staticmethod
    def calculateJacobian(variables: List[sp.Symbol],
                          parametrization: sp.Array, ) -> sp.Array:
        """
        This function calculates the Jacobian of a Riemannian manifold.
        :param variables:
        :param parametrization:
        :return:
        """
        # Calculate the Jacobian Matrix
        jacobiMatrix = sp.derive_by_array(parametrization, variables).transpose()
        return jacobiMatrix

    @staticmethod
    def calculateMetricTensor(variables: List[sp.Symbol],
                              parametrization: sp.Array, ) -> sp.Array:
        """
        This function calculates the metric tensor of a Riemannian manifold.
        :param variables:
        :param parametrization:
        :return:
        """
        # Calculate the Jacobian Matrix
        jacobiMatrix = RiemannianManifold.calculateJacobian(variables, parametrization)

        # Calculate the metric tensor
        def metricTensor(jM: sp.Matrix) -> sp.Matrix:
            return sp.simplify(jM.T * jM)

        return feedThroughArrMat(jacobiMatrix, metricTensor)

    @staticmethod
    def metricTensorDirect(jacobian: sp.Array) -> sp.Array:
        return feedThroughArrMat(jacobian, lambda jM: jM.T * jM)

    @staticmethod
    def recursive_inverse(Matrix: sp.Matrix) -> sp.Matrix:
        n = Matrix.shape[0]

        # Base case: 1x1 matrix
        if n == 1:
            return sp.Matrix([[1 / Matrix[0, 0]]])

        # Partition the matrix into A, b, and c
        A = Matrix[:-1, :-1]  # Top-left (n-1)x(n-1)
        b = Matrix[:-1, -1]  # Last column (n-1)x1 (as a vector)
        c = Matrix[-1, -1]   # Bottom-right scalar

        # Recursively compute A^-1
        A_inv = RiemannianManifold.recursive_inverse(A)

        # Compute k = c - b.T * A^-1 * b
        k = c - (b.T * A_inv * b)[0, 0]  # SymPy returns a matrix; extract scalar

        if k == 0:
            raise ValueError("Matrix is not invertible.")

        # Compute the full inverse
        top_left = A_inv + (1 / k) * (A_inv * b * b.T * A_inv)
        top_right = -(1 / k) * (A_inv * b)
        bottom_left = -(1 / k) * (b.T * A_inv)
        bottom_right = sp.Matrix([[1 / k]])

        # Assemble the full inverse
        top = sp.Matrix.hstack(top_left, top_right)
        bottom = sp.Matrix.hstack(bottom_left, bottom_right)
        return sp.Matrix.vstack(top, bottom)

    @staticmethod
    def calculateInverseMetricTensor(variables: List[sp.Symbol],
                                     parametrization: sp.Array,
                                     metricTensor: sp.Array = None) -> sp.Array:
        """
        This function calculates the inverse metric tensor of a Riemannian manifold.
        It now includes a test to compare the performance of SymPy and NumPy inversion methods.
        :param metricTensor:
        :param variables: List of symbolic variables.
        :param parametrization: Parametrization of the manifold.
        :return: Inverse metric tensor as a SymPy Array.
        """
        import time

        # Calculate the metric tensor
        metricTensor = RiemannianManifold.calculateMetricTensor(variables,
                                                                parametrization) if metricTensor is None else metricTensor

        # Convert the metric tensor to both SymPy Matrix and NumPy array
        numpy_array = np.array(metricTensor.tolist())
        sympy_matrix = sp.Matrix(metricTensor.tolist())

        # Time SymPy inversion
        start_time_sympy = time.perf_counter()
        sympy_inv = RiemannianManifold.recursive_inverse(sympy_matrix)
        end_time_sympy = time.perf_counter()
        sympy_time = end_time_sympy - start_time_sympy

        """print(f"SymPy inversion time: {sympy_time:.6f} seconds")

        # Time NumPy inversion
        start_time_numpy = time.perf_counter()
        numpy_inv = numpy_array.inv()
        end_time_numpy = time.perf_counter()
        numpy_time = end_time_numpy - start_time_numpy

        # Print inversion times

        print(f"NumPy inversion time: {numpy_time:.6f} seconds")

        # Optionally, compare inversion results
        if sympy_inv.tolist() == numpy_inv.tolist():
            print("Inversion results are identical.")
        else:
            print("Inversion results differ.")"""
        print(f"SymPy inversion time: {sympy_time:.6f} seconds")

        # Use the inversion method you prefer (here, we use SymPy's inversion)
        inverseMetricTensor = sympy_inv

        return sp.Array(inverseMetricTensor.tolist())

    @staticmethod
    def inverseMetricTensorDirect(metricTensor: sp.Array) -> sp.Array:
        return sp.Array(metricTensor.tomatrix().inv().tolist()).simplify()

    @staticmethod
    def calculateChristoffelSymbols(variables: List[sp.Symbol],
                                    parametrization: sp.Array, ) -> sp.Array:
        """
        This function calculates the Christoffel Symbols of a Riemannian manifold.
        :param variables:
        :param parametrization:
        :return:
        """
        """# Calculate the Christoffel Symbols
        J = np.array(RiemannianManifold.calculateJacobian(variables, parametrization))
        g = np.array(RiemannianManifold.calculateMetricTensor(variables, parametrization))
        g_inv = np.array(RiemannianManifold.calculateInverseMetricTensor(variables, parametrization))
        D_J = np.array(sp.derive_by_array(J, variables))
        # print(f"D_J: {D_J}")
        D_g = np.array(sp.derive_by_array(g, variables))
        # print(f"D_g: {D_g}")
        R = np.array(sp.Array(np.einsum("ijk,jl->ikl", D_J, J)).simplify())
        # print(f"R: {R}")
        f = (R + np.einsum("ijk->jik", R) + np.einsum("ijk->ikj", R) +
             np.einsum("ijk->jki", R))
        S = R - D_g
        Christoffel = np.einsum("ijk,jl->lik", -S, g_inv)
        Christoffel = np.array(sp.Array(Christoffel.tolist()).simplify())
        # print(f"Christoffel: {Christoffel}")
        return sp.Array(Christoffel.tolist())"""
        # Calculate the Jacobian Matrix
        jacobiMatrix = RiemannianManifold.calculateJacobian(variables, parametrization)
        # Calculate the metric tensor
        metricTensor = RiemannianManifold.metricTensorDirect(jacobiMatrix)
        # Calculate the inverse metric tensor
        inverseMetricTensor = RiemannianManifold.inverseMetricTensorDirect(metricTensor)
        # Calculate the Christoffel Symbols
        return RiemannianManifold.christoffelSymbolsDirect(variables, metricTensor, inverseMetricTensor)

    @staticmethod
    def christoffelSymbolsDirect(variables: List[sp.Symbol], metricTensor: sp.Array,
                                 invMT: sp.Array = None) -> sp.Array:
        if invMT is None:
            invMT = RiemannianManifold.inverseMetricTensorDirect(metricTensor)
        metricTensor = np.array(metricTensor)
        invMT = np.array(invMT)
        D_g = np.array(sp.derive_by_array(metricTensor, variables))
        D_g_ijk = D_g
        D_g_kij = np.einsum("ijk->kij", D_g)
        D_g_jki = np.einsum("ijk->jki", D_g)
        """ijkRef = np.array([[["uuu", "uuv", "uuz"], ["uvu", "uvv", "uvz"], ["uzu", "uzv", "uzz"]],
                           [["vuu", "vuv", "vuz"], ["vvu", "vvv", "vvz"], ["vzu", "vzv", "vzz"]],
                           [["zuu", "zuv", "zuz"], ["zvu", "zvv", "zvz"], ["zzu", "zzv", "zzz"]]])
        kijRef = np.einsum("ijk->kij", ijkRef)
        jkiRef = np.einsum("ijk->jki", ijkRef)

        print(f"D_g_ijk: {D_g_ijk}")
        print(f"D_g_kij: {D_g_kij}")
        print(f"D_g_jki: {D_g_jki}")
        print(f"ijkRef: {ijkRef}")
        print(f"kijRef: {kijRef}")
        print(f"jkiRef: {jkiRef}")
        RRef = np.char.add(np.char.add(np.char.add(np.char.add(ijkRef, ' + '), kijRef), ' - '), jkiRef)"""
        R = D_g_ijk + D_g_kij - D_g_jki
        # print(f"R: {RRef}")
        # print(f"R: {R}")
        Christoffel = (1 / 2) * np.einsum("ijk,kl->kij", R, invMT)
        # Christoffel = np.einsum("ijk->kij", Christoffel)
        return sp.Array(Christoffel.tolist()).simplify()

    @staticmethod
    def calculateTensors(variables: List[str] | List[sp.Symbol],
                         parametrization: List[str] | List[sp.Expr] | np.ndarray[sp.Expr] | sp.Array,
                         checker: bool = False, simplify: bool = False, latex: bool = False) -> List[sp.Array]:
        """
        This function calculates the metric tensor of a Riemannian manifold.
        :param variables:
        :param parametrization:
        :param checker:
        :param simplify:
        :param latex:
        :return:
        """
        if isinstance(parametrization, list):
            if isinstance(parametrization[0], str):
                params = paramToSympyArray(paramToSympy(variables, parametrization, checker, simplify, latex))
            elif isinstance(parametrization[0], sp.Expr):
                params = paramToSympyArray(parametrization)
            else:
                raise ValueError("Parametrization must be a list of strings or a list of sympy expressions.")
        elif isinstance(parametrization, np.ndarray):
            if isinstance(parametrization[0], sp.Expr):
                params = sp.Array(parametrization)
            else:
                raise ValueError("Parametrization must be a list of sympy expressions.")
        elif isinstance(parametrization, sp.Array):
            params = parametrization
        else:
            raise ValueError("Parametrization must be a list of strings or a list of sympy expressions.")
        # Calculate the Jacobian Matrix
        jacobiMatrix = RiemannianManifold.calculateJacobian(variables, params)
        # Calculate the metric tensor
        metricTensor = jacobiMatrix.T @ jacobiMatrix
        # Calculate the inverse metric tensor
        inverseMetricTensor = metricTensor.inv()
        # Calculate the Christoffel Symbols


# Testing
if __name__ == "__main__":
    # Variables
    var = ["r", "o", "u"]
    # var = ["u", "v", "z"]
    # var = ["x", "y", "z"]
    # var = ["u", "v", "w"]
    # Parameters
    param = ["r*sin(o)*cos(u)", "r*sin(o)*sin(u)", "r*cos(o)"]
    # param = ["a*cosh(u)*cos(v)", "a*sinh(u)*sin(v)", "z"]
    # param = ["cosh(x)*tanh(y)*e^z", "sinh(x)tanh(y)*ln(z)", "atan(z)*x*y"]
    """param = ["(a * sin(u) * cos(v) + b * cos(w)) * exp(u)",
             "(a * sin(u) * sin(v) + b * sin(w)) * exp(v)",
             "(a * cos(u) + c * w) * exp(w)"]"""
    # Riemannian Manifold
    M = RiemannianManifold(3, var, param, verbose=True)
    print("Variables: ")
    print(M.variables)
    print("Jacobian: ")
    print(M.calculateJacobian(M.variables, M.arrayParam))
    print("Metric Tensor: ")
    mt = M.calculateMetricTensor(M.variables, M.arrayParam)
    print(mt)
    print(sp.printing.mathematica_code(mt))
    print("Inverse Metric Tensor: ")
    invMT = M.calculateInverseMetricTensor(M.variables, M.arrayParam, mt)
    print(invMT)
    # Christoffel Symbols
    print("Christoffel Symbols: ")
    christ = M.calculateChristoffelSymbols(M.variables, M.arrayParam)
    print(christ)
    print(sp.printing.mathematica_code(christ))
