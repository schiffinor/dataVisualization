'''
linear_regression.py
Subclass of Analysis that performs linear regression on data
Roman Schiffino
CS251 Data Analysis Visualization
Spring 2024
'''
import time

import numpy
import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
import matplotlib.figure as fig__
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Operands import *
from typing import List, Dict
from regressTypes import *
from data import Data
import scipy.fft as sfft
import scipy.signal.windows as swa
from regressTypes import RegressTypes
import numpy.polynomial.polynomial as npp
from mpl_toolkits import mplot3d
from scipy import signal
import matplotlib.rcsetup as rcsetup
import analysis


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''
        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable (true values) being predicted by linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean SEE. float. Measure of quality of fit
        self.mse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        self.lin_opp = LeftOperand(lambda x: self.intercept + np.einsum('ij,ji->i', x, self.slope))
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var):
        '''
        Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression by using Scipy to solve the least squares problem y = Ac
        for the vector c of regression fit coefficients. Don't forget to add the coefficient column
        for the intercept!
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.y = self.data.select_data([dep_var])
        self.A = self.data.select_data(ind_vars)
        augA = np.hstack([np.ones((self.A.shape[0], 1)), self.A])
        c, residue, rank, s = sla.lstsq(augA, self.y)
        self.slope = c[1:]
        c = np.squeeze(c)
        self.intercept = c[0]
        predicts = self.predict()
        self.residuals = self.compute_residuals(predicts)
        self.R2 = self.r_squared(predicts)
        self.operand = LeftOperand(lambda x: self.intercept + np.einsum('ij,ji->i', x, self.slope))
        label_preComp = [f"{c[i]:.2f}x_{i}" for i in range(len(c))]
        label_comp = f"y = {c[0]}" + " + ".join(label_preComp)
        return self.operand, label_comp

    def predict(self, X=None):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''
        if X is None:
            X = self.A
        return (self.lin_opp | X)[:, np.newaxis]

    def r_squared(self, y_pred=None):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        return 1 - self.compute_sse(y_pred) / self.compute_sst()

    def compute_residuals(self, y_pred=None):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''
        if y_pred is None:
            y_pred = self.predict()
        return self.y - y_pred

    def compute_mse(self, y_pred=None):
        '''Computes the mean squared error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean squared error

        Hint: Make use of self.compute_residuals
        '''
        residuals = self.compute_residuals(y_pred)
        return 1 / len(self.y) * np.einsum('i,i->', residuals.flatten(), residuals.flatten())

    def compute_sse(self, y_pred=None):
        '''Computes the sum of squared errors in the predicted y compared the actual y values.

        Returns:
        -----------
        float. Sum of squared errors
        '''
        return self.compute_mse(y_pred) * len(self.y)

    def compute_sst(self):
        '''Computes the total sum of squares of the mean deviation in the variable y.

        Returns:
        -----------
        float. Total sum of squares
        '''
        return self.compute_smd() * len(self.y)

    def compute_ssr(self, y_pred=None):
        '''Computes the sum of squares due to regression.

        Returns:
        -----------
        float. Sum of squares due to regression
        '''
        return self.compute_sst() - self.compute_sse(y_pred)

    def compute_smd(self):
        '''Computes the squared mean deviation of the residuals.

        Returns:
        -----------
        float. Squared mean deviation
        '''
        mean_errors = self.compute_residuals(1 / len(self.y) * np.einsum('i->', self.y.flatten()))
        return 1 / len(self.y) * np.einsum('i,i->', mean_errors.flatten(), mean_errors.flatten())

    def scatter(self, ind_var: str | np.ndarray, dep_var: str | np.ndarray, title: str, operand: LeftOperand = None,
                R2: float = None, reg_label: str = "Regression", residuals_on: bool = False, residuals: np.ndarray = None,
                fig_sz=(14, 12), colors=None, alt_mode=False, fig: fig__.Figure = None, axes: plt.Axes = None):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''
        if operand is None:
            operand = self.lin_opp
        if colors is None:
            colors = ['blue', 'red']
        if isinstance(ind_var, str):
            x_data = self.data.select_data([ind_var])
        elif isinstance(ind_var, np.ndarray):
            x_data = ind_var
            if x_data.ndim < 2:
                x_data = x_data[:, np.newaxis]
            if x_data.ndim > 2:
                raise ValueError("ind_var must be 2ds")
            if x_data.shape[1] > 1:
                raise ValueError("ind_var must be a single column")
        else:
            raise ValueError("ind_var must be a string or a numpy array")
        if isinstance(dep_var, str):
            y_data = self.data.select_data([dep_var])
        elif isinstance(dep_var, np.ndarray):
            y_data = dep_var
            if y_data.ndim < 2:
                y_data = y_data[:, np.newaxis]
            if y_data.ndim > 2:
                raise ValueError("dep_var must be 2d")
            if y_data.shape[1] > 1:
                raise ValueError("dep_var must be a single column")
        else:
            raise ValueError("dep_var must be a string or a numpy array")
        sort = x_data.flatten().argsort()
        x_data = x_data[sort]
        y_data = y_data[sort]
        x_max = np.max(x_data)
        x_min = np.min(x_data)
        y_max = np.max(y_data)
        y_min = np.min(y_data)
        d_domain = x_max - x_min
        d_range = y_max - y_min
        line_x = np.linspace(x_min, x_max, 1000)[:, np.newaxis]
        line_y = operand | line_x
        if not alt_mode:
            figure, axis = plt.subplots(figsize=fig_sz, sharex=True)
        else:
            if fig is None:
                raise ValueError("fig must be provided")
            if axes is None:
                raise ValueError("index must be provided")
            figure = fig
            axis = axes
        divider = make_axes_locatable(axis)
        axis.scatter(x_data, y_data, color=colors[0])
        axis.plot(line_x, line_y, color=colors[1], label=reg_label)
        if not alt_mode:
            axis.set_title(title)
            splitter = title.split(' by ')
            if len(splitter) > 1:
                splitter = splitter[1].split(' ')
                axis.set_ylabel(splitter[0], rotation=0 if len(splitter[0]) < 5 else 90, labelpad=10)
            else:
                axis.set_xlabel('X')
                axis.set_ylabel('Y', labelpad=10, rotation=0)
        else:
            if R2 is not None:
                axis.set_title(f" R^2 = {R2:.2f}")
        if residuals_on:
            if residuals is None:
                predict = operand | x_data
                residuals = y_data - predict[:, np.newaxis]
            if not alt_mode:
                axis2 = divider.append_axes("bottom", size='25%', pad=0, yticks=np.linspace(np.min(residuals), np.max(residuals), 5))
            else:
                axis2 = divider.append_axes("bottom", size='25%', pad=0)
            figure.add_axes(axis2)
            axis2.vlines(x_data, 0, residuals, color='purple', alpha=0.5)
            axis2.axhline(0, color="teal")
            if not alt_mode:
                splitter = title.split(' by ')
                if len(splitter) > 1:
                    axis2.set_xlabel(splitter[0])
                else:
                    axis2.set_xlabel('X')
                axis2.set_ylabel("Residuals")

                legend = figure.legend(title="Regression", bbox_to_anchor=(0.07, 0.015), loc="lower left")
                legend._legend_box.align = "left"
                figure.subplots_adjust(bottom=0.15)
            return figure, axis, axis2
        if not alt_mode:
            legend = figure.legend(title="Regression", bbox_to_anchor=(0.07, 0.015), loc="lower left")
            legend._legend_box.align = "left"
            figure.subplots_adjust(bottom=0.15)
        return figure, axis

    def pair_plot(self, data_vars, fig_sz=(14, 14), hists_on_diag=True, reg_type: RegressTypes = RegressTypes.linear, degree=1, title=''):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''
        size = len(data_vars)
        figure = plt.figure(figsize=fig_sz, layout="compressed")
        axes = figure.subplots(size, size, sharex="col", sharey="row" if not hists_on_diag else False)
        for row, dep in enumerate(data_vars):
            for col, ind in enumerate(data_vars):
                place = axes[row, col]
                ind_data = self.data.select_data([ind])
                dep_data = self.data.select_data([dep])
                reg = self.regression(ind_data, dep_data, plot_on=True, regress_type=reg_type, degree=degree, alt_out=True)[2]
                if row == col and hists_on_diag:
                    place = axes[row, col]
                    place.hist(ind_data, bins=20, color='purple', alpha=0.5)
                    place_sub = place
                else:
                    index = 0 if row != 0 else size-1
                    ax1 = axes[row, index]
                    figures, place, place_sub = self.scatter(*reg[0], **reg[1], alt_mode=True, fig=figure, axes=place)
                    if col != index and hists_on_diag:
                        place.sharey(ax1)
                if hists_on_diag:
                    if col != row and (col != 0 and (row != 0 or col != 1)):
                        place.tick_params(labelleft=False)
                        place_sub.set_yticks([])
                elif col != 0 and (row != 0 or col != 1):
                    place_sub.set_yticks([])
                if row == size - 1:
                    place_sub.set_xlabel(ind)
                elif row != col or not hists_on_diag:
                    place_sub.set_xticks([])
                if col == 0:
                    place.set_ylabel(dep)
        figure.suptitle(title)

    @staticmethod
    def make_polynomial_matrix(A, degree):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        should take care of that.
        '''
        return np.column_stack([np.power(A, i) for i in range(degree + 1)])

    @staticmethod
    def comp_matrix(A: np.ndarray, degree: int):
        A_transpose = A.T
        mat_List = []
        index = 0
        for row in A_transpose.tolist():
            row_ = np.array(row)
            temp_augA = LinearRegression.make_polynomial_matrix(row_, degree)
            mat_List.append(temp_augA if index == 0 else temp_augA[:, 1:])
            index += 1
        augA = np.hstack(mat_List)
        return augA

    def poly_regression(self, ind_var: str, dep_var: str, degree: int):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and an added column of 1s for the intercept.

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create a matrix based on the independent variable data matrix (self.A) with columns
            appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        y_data = self.data.select_data([dep_var])
        A_data = self.data.select_data([ind_var])
        return self.multi_poly_regression(A_data, y_data, degree)

    @staticmethod
    def linear_regression_2(A: np.ndarray, y: np.ndarray):
        '''
        Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression by using Scipy to solve the least squares problem y = Ac
        for the vector c of regression fit coefficients. Don't forget to add the coefficient column
        for the intercept!
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''

        augA = np.hstack([np.ones((A.shape[0], 1)), A])
        c, residue, rank, s = sla.lstsq(augA, y)
        c = np.squeeze(c)
        slope = c[1:]
        intercept = c[0]
        operand = LeftOperand(lambda x: intercept + np.einsum('ij,j->i', x, slope))
        label_preComp = [f"{slope[i]:.2f}x_{i}" for i in range(len(slope))]
        label_comp = f"y = {intercept} + " + " + ".join(label_preComp)
        return operand, label_comp

    @staticmethod
    def multi_poly_regression(ind_vars: np.ndarray, dep_var: np.ndarray, degree: int):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and an added column of 1s for the intercept.

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create a matrix based on the independent variable data matrix (self.A) with columns
            appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        augA = LinearRegression.comp_matrix(ind_vars, degree)
        c, residue, rank, s = sla.lstsq(augA, dep_var)
        label_preComp = []
        if ind_vars.shape[1] == 1:
            label_preComp = [f"{c[i][0]:.2f}x^{i}" for i in range(1, degree + 1)]
        else:
            for j in range(1, ind_vars.shape[1] + 1):
                label_preComp += [f"{c[i][0]:.2f}x_{j}^{i}" for i in range(1, degree + 1)]
        label_comp = f"y = {c[0][0]}" + " + ".join(label_preComp)
        operand = LeftOperand(lambda x: np.einsum('ij,j->i', LinearRegression.comp_matrix(x, degree), np.squeeze(c)))
        return operand, label_comp

    @staticmethod
    def exponential_regression(ind_var: np.ndarray, dep_var: np.ndarray, degree: int = 1):
        ind_var = np.squeeze(ind_var)
        dep_var = np.squeeze(dep_var)
        if ind_var.shape != dep_var.shape:
            raise ValueError("ind_var and dep_var must have the same length")
        if ind_var.ndim > 1:
            raise ValueError("ind_var must be a single column")
        else:
            ind_shift = np.min(ind_var)
            ind_var = ind_var - ind_shift
            iy_list = [LinearRegression.d_int(ind_var, dep_var)]
            i_s = ["iy"]
            for i in range(degree - 1):
                iy_list.append(LinearRegression.d_int(ind_var, iy_list[-1]))
                i_s.append("i" + i_s[-1])
            augA = np.column_stack(iy_list)
            augA = np.hstack([augA, np.fliplr(LinearRegression.make_polynomial_matrix(ind_var, degree))])
            c = sla.lstsq(augA, dep_var)[0]
            c_ = (c[:degree][:, np.newaxis]).T
            augIdent = np.hstack([np.identity(degree - 1), np.zeros((degree - 1, 1))])
            eigA = np.vstack([c_, augIdent])
            lambdas = np.real(sla.eigvals(eigA))
            linear_component_list = [np.ones(ind_var.shape[0])] + [np.exp(lambda_ * (ind_var + ind_shift)) for lambda_ in lambdas]
            augA_2 = np.column_stack(linear_component_list) / dep_var[:, np.newaxis]
            mega_A = np.einsum('ij,ik->ijk', augA_2, augA_2)
            mega_dep = augA_2
            args = (~(np.logical_or(np.isnan(mega_A), np.isinf(mega_A))).any(axis=1)).all(axis=1)
            mega_A = mega_A[args, :, :]
            mega_dep = mega_dep[args, :]
            mega_A = np.einsum('ijk->jk', mega_A)
            mega_dep = np.einsum('ij->j', mega_dep)
            coeffs = sla.lstsq(mega_A, mega_dep)[0][np.newaxis, :]
            coeffs_ = coeffs.flatten().tolist()
            lambdas_ = lambdas.flatten().tolist()
            operand = LeftOperand(lambda x: np.sum(coeffs * np.column_stack([np.ones(x.shape[0])] + [np.exp(lambda_ * x) for lambda_ in lambdas]), axis=1))
            label_preComp = [f"{coeffs_[i + 1]:.2f}e^({lambdas_[i]: .2f}x)" for i in range(len(lambdas_))]
            label_comp = f"y = {coeffs_[0]:.2f} + " + " + ".join(label_preComp)
        return operand, label_comp

    @staticmethod
    def sinusoidal_regression(ind_var: np.ndarray, dep_var: np.ndarray, degree: int):
        ind_var = np.squeeze(ind_var)
        dep_var = np.squeeze(dep_var)
        if degree < 1:
            raise ValueError("Degree must be greater than 0")
        if degree > ind_var.shape[0] // 2:
            raise ValueError("Degree too high for sinusoidal regression!")
        if ind_var.shape != dep_var.shape:
            raise ValueError("ind_var and dep_var must have the same length")
        if ind_var.ndim > 1:
            raise ValueError("ind_var must be a single column")

        sort = ind_var.argsort()
        ind_var = ind_var[sort]
        dep_var = dep_var[sort]
        window = swa.gaussian(len(ind_var), len(ind_var))
        fft = sfft.rfft(dep_var)

        N = len(ind_var)
        n = np.linspace(ind_var.min(), ind_var.max(), len(ind_var))
        # get the sampling rate
        freq = sfft.rfftfreq(N, (n[1] - n[0]))

        # plt.figure(figsize=(12, 6))
        # plt.vlines(freq, 0, np.abs(fft), 'b')
        # plt.xlabel('Freq (Hz)')
        # plt.ylabel('FFT Amplitude |X(freq)|')

        fftMag = 2 * np.abs(fft) / np.sum(window)
        fftMag[0] = fftMag[0] / 2
        fftPhase = np.angle(fft)

        # Index of max
        maxIndex = np.argpartition(fftMag[0:N // 2 + 1], -degree)[-degree:]

        # Calculate the amplitude
        amps = fftMag[maxIndex]
        # Calculate the frequency
        maxFreq = freq[maxIndex]
        # Calculate the phase shift
        phaseShift = fftPhase[maxIndex] + np.pi / 2

        label_preComp = [
            f"{amps[i]:.2f}sin(2π*{maxFreq[i]: .2f}*x{f' + {phaseShift[i]: .2f}' if phaseShift[i] >= 0.01 else ''})"
            if maxFreq[i] != 0 else f"{amps[i]: .2f}"
            for i in range(len(amps))]
        label_comp = "y = " + " + ".join(label_preComp)
        operand = LeftOperand(lambda x: np.sum(amps * np.sin(2 * np.pi * maxFreq * x + phaseShift), axis=1))
        return operand, label_comp

    def regression(self, ind_vars_: List[str] | np.ndarray, dep_var_: str | np.ndarray,
                   regress_type: RegressTypes = RegressTypes.linear, degree: int = 1, plot_on: bool = False,
                   no_size: bool = False, alt_out=False):
        '''
        Perform a regression of type `regress_type` on the independent variables `ind_vars` and dependent variable `dep_var`.

        '''
        if regress_type not in RegressTypes.__members__:
            raise ValueError(f"regress_type must be one of {RegressTypes}")
        ind_vars = ind_vars_
        dep_var = dep_var_
        if isinstance(ind_vars, list):
            ind_count = len(ind_vars)
            if ind_count < 1:
                raise ValueError("Must have at least one independent variable")
            if any([not isinstance(i, str) for i in ind_vars]):
                raise ValueError("All independent variables must be strings")
            A = self.data.select_data(ind_vars)
        elif isinstance(ind_vars, np.ndarray):
            A = ind_vars
            if A.ndim < 2:
                A = A[:, np.newaxis]
            if A.ndim > 2:
                raise ValueError("ind_vars must be 2d.")
            ind_count = A.shape[1]
            if ind_count < 1:
                raise ValueError("Must have at least one independent variable")
            ind_vars = [f"x_{i}" for i in range(ind_count)]
        else:
            raise ValueError("ind_vars must be a list of strings or a numpy array")
        if isinstance(dep_var, str):
            y = self.data.select_data([dep_var])
        elif isinstance(dep_var, np.ndarray):
            y = dep_var
            if y.ndim < 2:
                y = y[:, np.newaxis]
            if y.ndim > 2:
                raise ValueError("dep_var must be 2d")
            if y.shape[1] > 1:
                raise ValueError("dep_var must be a single column")
            dep_var = "y"
        else:
            raise ValueError("dep_var must be a string or a numpy array")

        self.p = degree
        if regress_type == 'linear':
            self.ind_vars = ind_vars
            self.dep_var = dep_var
            self.y = y
            self.A = A
            opp, label = LinearRegression.linear_regression_2(A, y)
        elif regress_type == 'polynomial':
            opp, label = LinearRegression.multi_poly_regression(A, y, degree)
        elif regress_type == 'exponential':
            if ind_count > 1:
                raise ValueError("Exponential regression only supports one independent variable")
            opp, label = LinearRegression.exponential_regression(A, y, degree)
        elif regress_type == 'sinusoidal':
            if ind_count > 1:
                raise ValueError("Sinusoidal regression only supports one independent variable")
            opp, label = LinearRegression.sinusoidal_regression(A, y, degree)
        elif regress_type == 'mixed':
            print("Mixed regression treats only first independent variable as exponential and the rest as polynomial")
            raise ValueError("Mixed regression not yet implemented. Please use a different regression type.")
        else:
            raise ValueError(f"regress_type must be one of {RegressTypes}")
        sample_size = len(y)
        mean = 1 / sample_size * np.einsum('i->', y.flatten())
        predicts = (opp | A)[:, np.newaxis]
        residuals = y - predicts
        max_resid = np.max(residuals)
        min_val = np.min(y)
        mean_errors = y - mean
        smd = 1 / sample_size * np.einsum('i,i->', mean_errors.flatten(), mean_errors.flatten())
        mse = 1 / sample_size * np.einsum('i,i->', residuals.flatten(), residuals.flatten())
        sse = mse * sample_size
        sst = smd * sample_size
        ssr = sst - sse
        R2 = 1 - (sse / sst)
        out = None
        if plot_on:
            if ind_count == 1:
                args = [A[:, 0], y, f"{ind_vars[0]} by {dep_var} with {regress_type} Regression (R^2 = {R2:.2f})"]
                kwargs = {"operand": opp, "reg_label": label, "residuals_on": True, "fig_sz": None, "R2": R2}
                if alt_out:
                    out = [args, kwargs]
                else:
                    out = self.scatter(*args, **kwargs)
        return opp, label, out



    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        return self.intercept

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.

        TODO:
        - Use parameters and call methods to set all instance variables defined in constructor. 
        '''
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.slope = slope
        self.intercept = intercept
        self.p = p

    @staticmethod
    def d_derivative(x: np.ndarray, y: np.ndarray):
        '''
        Calculates an approximate derivative of the data,
        :param self:
        :param x:
        :param y:
        :return:
        '''
        if x.shape != y.shape:
            raise ValueError("x and y must have the same length")
        if x.shape[0] < 2:
            raise ValueError("x and y must have at least 2 elements")
        arg = x.argsort()
        x_mod = x[arg]
        y_mod = y[arg]
        mod_mat_x = np.append(x_mod, np.nan)
        mod_mat_y = np.append(y_mod, np.nan)
        comp_mat_x_up = np.column_stack([mod_mat_x, -np.roll(mod_mat_x, -1)])
        comp_mat_x_down = np.column_stack([-mod_mat_x, np.roll(mod_mat_x, 1)])
        comp_mat_y_up = np.column_stack([mod_mat_y, -np.roll(mod_mat_y, -1)])
        comp_mat_y_down = np.column_stack([-mod_mat_y, np.roll(mod_mat_y, 1)])
        dx_mat = np.column_stack([np.einsum('ij->i', comp_mat_x_up), np.einsum('ij->i', comp_mat_x_down)])[:-1]
        dy_mat = np.column_stack([np.einsum('ij->i', comp_mat_y_up), np.einsum('ij->i', comp_mat_y_down)])[:-1]
        dy_dx_d_mat = dy_mat / dx_mat
        dy_dx_mat = np.nanmean(dy_dx_d_mat, axis=1)
        return dy_dx_mat

    @staticmethod
    def d_int(x: np.ndarray, y: np.ndarray):
        '''
        Calculates an approximate derivative of the data,
        :param self:
        :param x:
        :param y:
        :return:
        '''
        if x.shape != y.shape:
            raise ValueError("x and y must have the same length")
        if x.shape[0] < 2:
            raise ValueError("x and y must have at least 2 elements")
        arg = x.argsort()
        x_mod = x[arg]
        y_mod = y[arg]
        comp_mat_x_up = np.column_stack([-x_mod, np.roll(x_mod, -1)])[:-1]
        comp_mat_y_up = np.column_stack([-y_mod, np.roll(y_mod, -1)])[:-1]
        dx_mat = np.array(np.einsum('ij->i', comp_mat_x_up)).tolist()
        dy_mat = np.array(np.einsum('ij->i', comp_mat_y_up)).tolist()
        accum = [0]
        for i in range(x.shape[0] - 1):
            accum.append(dx_mat[i] * (y_mod[i] + (1 / 2) * dy_mat[i]) + accum[-1])
        cum_sum = np.array(accum)
        return cum_sum


if __name__ == '__main__':

    data = Data('data/iris.csv')
    analysis = LinearRegression(data)
    # Create a simple linear function y = 2x^2 + 3
    x_ = np.linspace(0, 10, 500)
    y_ = 2 * x_ ** 2 + 3

    '''integral = LinearRegression.d_int(x_, y_)

    # Since y = 2x^2 + 3 is a linear function, its integral is (2/3)x^3 + 3x + C
    # Let's compare the calculated integral with the expected integral
    expected_integral = (2 / 3) * x_ ** 3 + 3 * x_ + 0
    print("Calculated integral: ", integral)
    print("Expected integral: ", expected_integral)
    np.testing.assert_allclose(integral, expected_integral, rtol=1e-4)

    print("Integral tests passed!")
    '''
    '''# Calculate derivative
    derivative = analysis.d_derivative(x_, y_)

    # Since y = 2x^2 + 3 is a linear function, its derivative is 4x
    # Let's compare the calculated derivative with the expected derivative
    expected_derivative = 4 * x_
    print("Calculated derivative: ", derivative)
    print("Expected derivative: ", expected_derivative)
    np.testing.assert_allclose(derivative[1:-1], expected_derivative[1:-1], rtol=1e-4)
    '''

    # Perform sinusoidal regression
    data = np.genfromtxt("Lab03a/data/mystery_data_1.csv", delimiter=",")[2:]


    x_dat = data[:, 0]
    y_dat = data[:, 1]
    analysis.regression(x_dat, y_dat, RegressTypes.sinusoidal, 1, plot_on=True)

    x_m = np.linspace(0, 4 * np.pi, 5000)
    y_m = 2 * np.sin(x_m) + 3 * np.sin(2 * x_m) + 4 * np.sin(3 * x_m) + 5 + np.random.uniform(-0.5, 0.5, 5000)
    analysis.regression(x_m, y_m, RegressTypes.sinusoidal, 4, plot_on=True)
    '''oper, lab = analysis.sinusoidal_regression(x_m, y_m, 4)
    plt.plot(x_m, (oper | x_m[:, np.newaxis]), "r", label=lab)
    plt.scatter(x_m, y_m)
    plt.xlabel("X")
    plt.ylabel("Y", rotation=0, labelpad=10)
    plt.legend()
    plt.show()'''

    '''# Get fitted slope and intercept
    x_mm = np.linspace(-10, 10, 100)
    z_mm = np.linspace(-10, 10, 100)
    y_mm = x_mm ** 2 + np.random.uniform(-0.5, 0.5, 100)
    oper, lab = analysis.multi_poly_regression(x_mm[:, np.newaxis], y_mm[:, np.newaxis], 2)
    print(f"x_m: {x_mm[:, np.newaxis]}")
    nope = x_mm[:, np.newaxis]
    print(f"Operand: {oper | nope}")
    plt.plot(x_mm, oper | x_mm[:, np.newaxis], "r", label=lab)
    plt.scatter(x_mm, y_mm)
    plt.legend()
    plt.show()'''


    '''y_mm = x_mm ** 2 + z_mm * (z_mm - 5) * (z_mm + 5) + np.random.uniform(-0.5, 0.5, 100)
    zloperand = np.column_stack([x_mm[:, np.newaxis], z_mm[:, np.newaxis]])
    oper, lab = analysis.multi_poly_regression(zloperand, y_mm[:, np.newaxis], 3)
    plt.close()
    fig, ax = plt.subplots(figsize=(20, 10), subplot_kw={"projection": '3d'})
    ax.scatter3D(x_mm, z_mm, y_mm)
    ax.plot3D(x_mm, z_mm, np.squeeze(oper | zloperand), "r", label=lab)
    fig.show()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()'''
    """
    for angle in range(0, 360):
        ax.view_init(angle, 30)
        fig.canvas.draw()
        ax.draw(renderer)
        plt.pause(.001)
    for angle in range(0, 360):
        ax.view_init(0, angle)
        fig.canvas.draw()
        ax.draw(renderer)
        plt.pause(.001)
    """

    # fig.show()

    # Perform Exponential regression
    x_m = np.linspace(-10, 10, 5000)
    y_m = (30 + 2 * np.exp(3 * x_m) + 4 * np.exp(2 * x_m) + 6 * np.exp(4 * x_m) + np.random.uniform(-5, 5, 5000))
    analysis.regression(x_m, y_m, RegressTypes.exponential, 3, plot_on=True)
    '''# + np.random.uniform(-0.5, 0.5, 5000)
    oper, lab = LinearRegression.exponential_regression(x_m, y_m, 3)
    print(f"Operand: {oper | x_m}")
    plt.plot(x_m, oper | x_m[:, np.newaxis], "r", scaley=True)
    plt.scatter(x_m, y_m)
    plt.xlabel("X")
    plt.ylabel("Y", rotation=0, labelpad=10)
    plt.legend()
    plt.show()
    # Get fitted slope and intercept'''
    print("All tests passed!")
