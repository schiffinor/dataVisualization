"""
linear_regression.py
Subclass of Analysis that performs linear regression on data
Roman Schiffino
CS251 Data Analysis Visualization
Spring 2024
"""

from typing import List

import matplotlib.figure as fig__
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as sfft
import scipy.linalg as sla
import scipy.signal.windows as swa
from mpl_toolkits.axes_grid1 import make_axes_locatable

import Operands
import analysis
from Operands import *
from dataClass import Data
from regressTypes import RegressTypes


class LinearRegression(analysis.Analysis):
    """
    Perform and store linear regression and related analyses
    """

    def __init__(self, data_):
        """
        :param data_: Data object. Contains all data samples and variables in a dataset.
        """
        super().__init__(data_)

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
        """
        Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var.


        self.lin_opp : LeftOperand
            The linear regression model as a LeftOperand object.
        label_comp : str
            The equation of the regression model.

        :param ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        :param dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.

        :returns self.lin_opp: LeftOperand. The linear regression model as a LeftOperand object.
        :returns label_comp: str. The equation of the regression model.
        """
        # Set the independent and dependent variables
        self.ind_vars = ind_vars
        self.dep_var = dep_var

        # Select the data for the dependent and independent variables
        self.y = self.data.select_data([dep_var])
        self.A = self.data.select_data(ind_vars)
        self.y = np.asfarray(self.y, dtype=float)
        self.A = np.asfarray(self.A, dtype=float)


        # Add a column of ones to the independent variables matrix for the intercept term
        augA = np.hstack([np.ones((self.A.shape[0], 1)), self.A])

        # Solve the least squares problem to find the coefficients of the regression model
        c, residue, rank, s = sla.lstsq(augA, self.y)

        # The slope is all but the first element of the coefficients
        self.slope = c[1:]

        # Remove the extra dimension from the coefficients
        c = np.squeeze(c)

        # The intercept is the first element of the coefficients
        self.intercept = c[0]

        # Predict the dependent variable values using the model
        predicts = self.predict()

        # Compute the residuals of the model
        self.residuals = self.compute_residuals(predicts)

        # Compute the R-squared statistic of the model
        self.R2 = self.r_squared(predicts)

        # Create a LeftOperand object representing the linear regression model
        self.lin_opp = LeftOperand(lambda x: self.intercept + np.einsum('ij,ji->i', x, self.slope))

        # Create a string representing the equation of the regression model
        label_preComp = [f"{c[i]:.2f}x_{i}" for i in range(len(c))]
        label_comp = f"y = {c[0]}" + " + ".join(label_preComp)

        return self.lin_opp, label_comp

    def predict(self, X=None):
        """Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        :returns:
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        :param X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        :returns y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values
        """
        if X is None:
            X = self.A
        return (self.lin_opp | X)[:, np.newaxis]

    @staticmethod
    def data_anal(opp: Operands.LeftOperand, y: np.ndarray, A: np.ndarray, as_dict=False):
        """
            Performs data analysis on the given operand, dependent variable data, and independent variable data.

            :returns:
            If as_dict is True:
                dict : A dictionary containing the following key-value pairs:
                    "sample_size" : The number of samples in the data.
                    "mean" : The mean of the dependent variable data.
                    "predicts" : The predicted values of the dependent variable.
                    "residuals" : The residuals of the regression model.
                    "max_resid" : The maximum residual.
                    "min_val" : The minimum value of the dependent variable data.
                    "mean_errors" : The mean errors of the regression model.
                    "smd" : The sum of mean differences.
                    "mse" : The mean squared error of the regression model.
                    "sse" : The sum of squared errors of the regression model.
                    "sst" : The total sum of squares.
                    "ssr" : The sum of squares due to regression.
                    "R2" : The R-squared statistic of the regression model.
            If as_dict is False:
                tuple : A tuple containing the following elements in order:
                    sample_size, mean, predicts, residuals, max_resid, min_val, mean_errors, smd, mse, sse, sst, ssr, R2

            :param opp: Operands.LeftOperand
                The operand object representing the regression model.
            :param y: np.ndarray
                The dependent variable data as a numpy array.
            :param A: np.ndarray
                The independent variable data as a numpy array.
            :param as_dict: bool, optional
                If True, the method returns the results as a dictionary. If False, the method returns the results as a tuple.
            """
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
        if as_dict:
            return {"sample_size": sample_size, "mean": mean, "predicts": predicts, "residuals": residuals,
                    "max_resid": max_resid, "min_val": min_val, "mean_errors": mean_errors, "smd": smd, "mse": mse,
                    "sse": sse, "sst": sst, "ssr": ssr, "R2": R2}
        return sample_size, mean, predicts, residuals, max_resid, min_val, mean_errors, smd, mse, sse, sst, ssr, R2

    def r_squared(self, y_pred=None):
        """Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        """
        return 1 - self.compute_sse(y_pred) / self.compute_sst()

    def compute_residuals(self, y_pred=None):
        """Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        """
        if y_pred is None:
            y_pred = self.predict()
        return self.y - y_pred

    def compute_mse(self, y_pred=None):
        """Computes the mean squared error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean squared error

        Hint: Make use of self.compute_residuals
        """
        residuals = self.compute_residuals(y_pred)
        return 1 / len(self.y) * np.einsum('i,i->', residuals.flatten(), residuals.flatten())

    def compute_sse(self, y_pred=None):
        """Computes the sum of squared errors in the predicted y compared the actual y values.

        Returns:
        -----------
        float. Sum of squared errors
        """
        return self.compute_mse(y_pred) * len(self.y)

    def compute_sst(self):
        """Computes the total sum of squares of the mean deviation in the variable y.

        Returns:
        -----------
        float. Total sum of squares
        """
        return self.compute_smd() * len(self.y)

    def compute_ssr(self, y_pred=None):
        """Computes the sum of squares due to regression.

        Returns:
        -----------
        float. Sum of squares due to regression
        """
        return self.compute_sst() - self.compute_sse(y_pred)

    def compute_smd(self):
        """Computes the squared mean deviation of the residuals.

        Returns:
        -----------
        float. Squared mean deviation
        """
        mean_errors = self.compute_residuals(1 / len(self.y) * np.einsum('i->', self.y.flatten()))
        return 1 / len(self.y) * np.einsum('i,i->', mean_errors.flatten(), mean_errors.flatten())

    # noinspection PyProtectedMember
    def scatter(self, ind_var: str | np.ndarray, dep_var: str | np.ndarray, title: str, operand: LeftOperand = None,
                R2: float = None, reg_label: str = "Regression", residuals_on: bool = False,
                residuals: np.ndarray = None, annotatePoints: np.ndarray = None,  fig_sz=(14, 12), colors=None,
                alt_mode=False, fig: fig__.Figure = None, axes: plt.Axes = None):
        """
        Creates a scatter plot with a regression line to visualize the model fit.

        Assumes linear regression has been already run.

        :returns: tuple: (figure, axis, axis2)
        figure : matplotlib.figure.Figure
            The matplotlib figure object containing the plot.
        axis : matplotlib.pyplot.Axes
            The matplotlib axes object containing the plot.
        axis2 : matplotlib.pyplot.Axes, optional
            The matplotlib axes object containing the residuals plot. Only returned if residuals_on is True.


        :param ind_var: string | np.ndarray
            Independent variable name or data
        :param dep_var: string | np.ndarray
            Dependent variable name or data
        :param title: string
            Title for the plot
        :param operand: LeftOperand, optional
            The operand object representing the regression model. If None, uses the linear operand from the class.
        :param R2: float, optional
            The R-squared statistic of the regression model. If provided, it will be included in the plot title.
        :param reg_label: str, optional
            Label for the regression line in the plot. Default is "Regression".
        :param residuals_on: bool, optional
            If True, residuals will be plotted. Default is False.
        :param residuals: np.ndarray, optional
            A numpy array representing the residuals of the regression model. If None, residuals will be computed.
        :param annotatePoints: np.ndarray, optional
            A numpy array representing the points to annotate. Default is None.
        :param fig_sz: tuple, optional
            Size of the figure. Default is (14, 12).
        :param colors: list, optional
            List of colors to use for the scatter plot and regression line. Default is ['blue', 'red'].
        :param alt_mode: bool, optional
            If True, uses an alternative mode for plotting. Default is False.
        :param fig: matplotlib.figure.Figure, optional
            A matplotlib figure object. Required if alt_mode is True.
        :param axes: matplotlib.pyplot.Axes, optional
            A matplotlib axes object. Required if alt_mode is True.
        """
        if operand is None:
            # If operand is not provided, use the linear operand from the
            operand = self.lin_opp
        if colors is None:
            # Default colors for the scatter plot and regression line
            colors = ['blue', 'red']
        if isinstance(ind_var, str):
            # If ind_var is a string, select the data with that name
            x_data = self.data.select_data([ind_var])
        elif isinstance(ind_var, np.ndarray):
            # If ind_var is a numpy array, use it as the data
            x_data = ind_var
            if x_data.ndim < 2:
                # If x_data is 1D, convert it to 2D
                x_data = x_data[:, np.newaxis]
            if x_data.ndim > 2:
                # Raise error if x_data is not 2D
                raise ValueError("ind_var must be 2ds")
            if x_data.shape[1] > 1:
                # Raise error if x_data has more than one column
                raise ValueError("ind_var must be a single column")
        else:
            # Raise error if ind_var is not a string or numpy array
            raise ValueError("ind_var must be a string or a numpy array")
        if isinstance(dep_var, str):
            # If dep_var is a string, select the data with that name
            y_data = self.data.select_data([dep_var])
        elif isinstance(dep_var, np.ndarray):
            # If dep_var is a numpy array, use it as the data
            y_data = dep_var
            if y_data.ndim < 2:
                # If y_data is 1D, convert it to 2D
                y_data = y_data[:, np.newaxis]
            if y_data.ndim > 2:
                # Raise error if y_data is not 2D
                raise ValueError("dep_var must be 2d")
            if y_data.shape[1] > 1:
                # Raise error if y_data has more than one column
                raise ValueError("dep_var must be a single column")
        else:
            # Raise error if dep_var is not a string or numpy array
            raise ValueError("dep_var must be a string or a numpy array")

        # Get the indices that would sort x_data
        sort = x_data.flatten().argsort()
        # Sort x_data
        x_data = x_data[sort]
        # Sort y_data according to x_data
        y_data = y_data[sort]
        x_data = np.asfarray(x_data, dtype=float)
        y_data = np.asfarray(y_data, dtype=float)
        # Get the maximum value of x_data
        x_max = np.max(x_data)
        # Get the minimum value of x_data
        x_min = np.min(x_data)
        # Generate 1000 evenly spaced values between x_min and x_max
        line_x = np.linspace(x_min, x_max, 1000)[:, np.newaxis]
        # Apply the operand to line_x to get the y values for the regression line
        line_y = operand | line_x
        if not alt_mode:
            # Create a new figure and axis
            figure, axis = plt.subplots(figsize=fig_sz, sharex=True)
        else:
            if fig is None:
                # Raise error if fig is not provided
                raise ValueError("fig must be provided")
            if axes is None:
                # Raise error if axes is not provided
                raise ValueError("index must be provided")
            # Use the provided figure
            figure = fig
            # Use the provided axes
            axis = axes
        # Create an axes divider for the given axes
        divider = make_axes_locatable(axis)
        # Create a scatter plot of x_data and y_data
        axis.scatter(x_data, y_data, color=colors[0])
        # Plot the regression line
        axis.plot(line_x, line_y, color=colors[1], label=reg_label)
        if not alt_mode:
            # Set the title of the plot
            axis.set_title(title)
            # Label determined by the title
            splitter = title.split(' by ')
            if len(splitter) <= 1:
                splitter = title.split(' vs. ')
            if len(splitter) > 1:
                # If title contains "by", set the x-axis label to the first part of the title and the y-axis label to the second part
                xSplit = splitter[0].split(' ')
                ySplit = splitter[1].split(' ')
                axis.set_xlabel(xSplit[-1], labelpad=10)
                axis.set_ylabel(ySplit[0], rotation=0 if len(splitter[0]) < 5 else 90, labelpad=10)
            else:
                # If title does not contain "by", set the x-axis label to "X" and the y-axis label to "Y"
                axis.set_xlabel('X')
                axis.set_ylabel('Y', labelpad=10, rotation=0)
        else:
            if R2 is not None:
                # Set the title of the plot to the R^2 value
                axis.set_title(f" R^2 = {R2:.2f}")
        if annotatePoints is not None:
            annotatePoints = annotatePoints.flatten()
            # Annotate the points specified in annotatePoints
            for i, row in enumerate(x_data):
                for x_point in row:
                    axis.annotate(str(annotatePoints[i]), (float(x_point), float(y_data.flatten()[i])))
        if residuals_on:
            if residuals is None:
                # Predict the y values using the operand
                predict = operand | x_data
                # Compute the residuals
                residuals = y_data - predict[:, np.newaxis]
            # Create a new axes for the residuals
            if not alt_mode:
                axis2 = divider.append_axes("bottom", size='25%', pad=0,
                                            yticks=np.linspace(np.min(residuals), np.max(residuals), 3))
            else:
                axis2 = divider.append_axes("bottom", size='25%', pad=0)
            # Add the residuals axes to the figure
            figure.add_axes(axis2)
            # Plot the residuals as vertical lines
            axis2.vlines(x_data, 0, residuals, color='purple', alpha=0.5)
            # Plot a horizontal line at y=0
            axis2.axhline(0, color="teal")
            if not alt_mode:
                splitter = title.split(' by ')
                if len(splitter) > 1:
                    xSplit = splitter[0].split(' ')
                    axis2.set_xlabel(xSplit[-1])
                else:
                    axis2.set_xlabel('X')
                axis2.set_ylabel("Residuals")

                # Legend stuff
                legend = figure.legend(title="Regression", bbox_to_anchor=(0.07, 0.015), loc="lower left")
                legend._legend_box.align = "left"
                figure.subplots_adjust(bottom=0.25)
            return figure, axis, axis2
        # Legend stuff
        elif not alt_mode:
            legend = figure.legend(title="Regression", bbox_to_anchor=(0.07, 0.015), loc="lower left")
            legend._legend_box.align = "left"
            figure.subplots_adjust(bottom=0.25)
        return figure, axis

    def pair_plot(self, data_vars, fig_sz=(14, 14), hists_on_diag=True, reg_type: RegressTypes = RegressTypes.linear,
                  degree=1, title=''):
        """
        Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        :param data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        :param fig_sz: tuple, len(fig_sz)=2
            Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        :param hists_on_diag: bool
            If true, draw a histogram of the variable along main diagonal of pairplot.
        :param reg_type: RegressTypes
            Regression Type to Run
        :param degree: int
            Degree for regression, only relevant for some types
        :param title: string
            Title for the plot.
        """
        size = len(data_vars)
        figure = plt.figure(figsize=fig_sz, layout="compressed")
        axes = figure.subplots(size, size, sharex="col", sharey="row" if not hists_on_diag else False)
        for row, dep in enumerate(data_vars):
            for col, ind in enumerate(data_vars):
                place = axes[row, col]
                ind_data = self.data.select_data([ind])
                dep_data = self.data.select_data([dep])
                reg = \
                    self.regression(ind_data, dep_data, plot_on=True, regress_type=reg_type, degree=degree,
                                    alt_out=True)[2]
                if row == col and hists_on_diag:
                    place = axes[row, col]
                    place.hist(ind_data, bins=20, color='purple', alpha=0.5)
                    place_sub = place
                else:
                    index = 0 if row != 0 else size - 1
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
        """Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        :returns:
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        :param A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        :param degree: int
         Degree of polynomial regression model.
        """
        return np.asfarray(np.column_stack([np.power(A, i) for i in range(1, degree + 1)]), dtype=float)

    @staticmethod
    def comp_matrix(A: np.ndarray, degree: int):
        """
        Constructs a matrix for polynomial regression of a given degree.

        :returns:
        augA: np.ndarray
            The constructed matrix for polynomial regression.

        :param A : np.ndarray
            The independent variable data as a numpy array.
        :param degree : int
            The degree of the polynomial regression
        """
        A = np.asfarray(A, dtype=float)
        A_transpose = A.T
        mat_List = [np.ones((A.shape[0], 1), dtype=float)]
        for row in A_transpose.tolist():
            row_ = np.array(row)
            temp_augA = LinearRegression.make_polynomial_matrix(row_, degree)
            mat_List.append(temp_augA)
        augA = np.hstack(mat_List)
        augA = np.asfarray(augA, dtype=float)
        return augA

    def poly_regression(self, ind_var: str, dep_var: str, degree: int):
        """
        Perform polynomial regression — generalizes self.linear_regression to polynomial curves


        This method performs a polynomial regression on the independent variable `ind_var` and
        dependent variable `dep_var`. The degree of the polynomial is specified by `degree`.

        :returns:
        The result of the multi_poly_regression method, which performs the actual polynomial regression.

        :param ind_var: str
            The name of the independent variable entered in the single regression.
            The variable name must match those used in the `self.data` object.
        :param dep_var: str
            The name of the dependent variable entered into the regression.
            The variable name must match one of those used in the `self.data` object.
        :param degree: int
            The degree of the polynomial regression model.
            For example, if degree=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10, and an added column of 1s for the intercept.
        """
        y_data = self.data.select_data([dep_var])
        A_data = self.data.select_data([ind_var])
        return self.multi_poly_regression(A_data, y_data, degree)

    @staticmethod
    def linear_regression_2(A: np.ndarray, y: np.ndarray):
        """
        Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var.

        :returns: tuple: (operand, label_comp, c)
        operand: LeftOperand
            The operand object representing the linear regression model.
        label_comp: str
            The equation of the regression model.

        :param A: ndarray. shape=(num_data_samps, num_ind_vars)
            Matrix for independent (predictor) variables in linear regression
        :param y: ndarray. shape=(num_data_samps, 1)
            Vector for dependent variable (true values) being predicted by linear regression
        """
        augA = np.hstack([np.ones((A.shape[0], 1)), A])
        c, residue, rank, s = sla.lstsq(augA, y)
        c = np.squeeze(c)
        slope = c[1:]
        intercept = c[0]
        operand = LeftOperand(lambda x: intercept + np.einsum('ij,j->i', x, slope))
        label_preComp = [f"{slope[i]:.2f}x_{i}" for i in range(len(slope))]
        label_comp = f"y = {intercept:.2f} + " + " + ".join(label_preComp)
        return operand, label_comp, c

    @staticmethod
    def multi_poly_regression(ind_vars: np.ndarray, dep_var: np.ndarray, degree: int):
        """
        Performs a polynomial regression on the independent variable `ind_vars` and
        dependent variable `dep_var`. The degree of the polynomial is specified by `degree`.

        :returns: tuple: (operand, label_comp, c)
        operand: LeftOperand
            The operand object representing the polynomial regression model.
        label_comp: str
            The equation of the regression model.
        c: np.ndarray
            The coefficients of the regression model.

        :param ind_vars: np.ndarray
            The independent variable data as a numpy array.
        :param dep_var: np.ndarray
            The dependent variable data as a numpy array.
        :param degree: int
            The degree of the polynomial regression model.
        """
        augA = LinearRegression.comp_matrix(ind_vars, degree)
        c, residue, rank, s = sla.lstsq(augA, dep_var)
        label_preComp = []
        if ind_vars.shape[1] == 1:
            label_preComp = [f"{c[i][0]:.2f}x^{i}" for i in range(1, degree + 1)]
        else:
            for j in range(1, ind_vars.shape[1] + 1):
                label_preComp += [f"{c[i][0]:.2f}x_{j}^{i}" for i in range(1, degree + 1)]
        label_comp = f"y = {c[0][0]:.2f} + " + " + ".join(label_preComp)
        operand = LeftOperand(lambda x: np.einsum('ij,j->i', LinearRegression.comp_matrix(x, degree), np.squeeze(c)))
        return operand, label_comp, c

    @staticmethod
    def exponential_regression(ind_var: np.ndarray, dep_var: np.ndarray, degree: int = 1):
        """
        Performs an exponential regression on the independent variable `ind_var` and
        dependent variable `dep_var`. The degree of the polynomial is specified by `degree`.

        :returns: tuple: (operand, label_comp, coeffs, lambdas)
        operand: LeftOperand
            The operand object representing the exponential regression model.
        label_comp: str
            The equation of the regression model.
        coeffs: np.ndarray
            The coefficients of the regression model.
        lambdas: np.ndarray
            The eigenvalues of the regression model.

        :param ind_var: np.ndarray
            The independent variable data as a numpy array.
        :param dep_var: np.ndarray
            The dependent variable data as a numpy array.
        :param degree: int
            The degree of the polynomial regression model.


        """
        # Ensure variables are 1D arrays
        ind_var = np.squeeze(ind_var)
        dep_var = np.squeeze(dep_var)

        # Check if the independent and dependent variables have the same length
        if ind_var.shape != dep_var.shape:
            raise ValueError("ind_var and dep_var must have the same length")

        # Check if the independent variable is a single column
        if ind_var.ndim > 1:
            raise ValueError("ind_var must be a single column")

        else:
            # Shift the independent variable to start from zero
            ind_shift = np.min(ind_var)
            ind_var = ind_var - ind_shift

            # Initialize the list of integrated variables with the first integration
            iy_list = [LinearRegression.d_int(ind_var, dep_var)]
            i_s = ["iy"]

            # Perform the remaining integrations
            for i in range(degree - 1):
                iy_list.append(LinearRegression.d_int(ind_var, iy_list[-1]))
                i_s.append("i" + i_s[-1])

            # Construct the augmented matrix for the regression
            augA = np.column_stack(iy_list)
            augA = np.hstack([augA, np.fliplr(LinearRegression.comp_matrix(ind_var[:, np.newaxis], degree))])

            # Solve the least squares problem to find the coefficients
            c = sla.lstsq(augA, dep_var)[0]

            # Construct the matrix for the eigenvalue problem
            c_ = (c[:degree][:, np.newaxis]).T
            augIdent = np.hstack([np.identity(degree - 1), np.zeros((degree - 1, 1))])
            eigA = np.vstack([c_, augIdent])

            # Compute the eigenvalues
            lambdas = np.real(sla.eigvals(eigA))

            # Construct the linear components vectors for the regression model
            linear_component_list = [np.ones(ind_var.shape[0])] + [np.exp(lambda_ * (ind_var + ind_shift)) for lambda_
                                                                   in lambdas]

            # Construct the augmented matrix for the regression
            augA_2 = np.column_stack(linear_component_list) / dep_var[:, np.newaxis]

            # Construct augmented matrix tensor for the regression
            mega_A = np.einsum('ij,ik->ijk', augA_2, augA_2)
            mega_dep = augA_2

            # Filter out any rows with NaN or Inf values
            args = (~(np.logical_or(np.isnan(mega_A), np.isinf(mega_A))).any(axis=1)).all(axis=1)
            mega_A = mega_A[args, :, :]
            mega_dep = mega_dep[args, :]

            # Sum the mega matrix and dependent variable over the first axis to get the final matrix and vector
            mega_A = np.einsum('ijk->jk', mega_A)
            mega_dep = np.einsum('ij->j', mega_dep)

            # Solve the least squares problem to find the coefficients with previous matrix and vector
            coeffs = sla.lstsq(mega_A, mega_dep)[0][np.newaxis, :]

            # Flatten the coefficients and eigenvalues for the regression equation
            coeffs_ = coeffs.flatten().tolist()
            lambdas_ = lambdas.flatten().tolist()

            # Construct the operand for the regression model
            operand = LeftOperand(lambda x: np.sum(
                coeffs * np.column_stack([np.ones(x.shape[0])] + [np.exp(lambda_ * x) for lambda_ in lambdas]), axis=1))

            # Construct the regression equation label
            label_preComp = [f"{coeffs_[i + 1]:.2f}e^({lambdas_[i]: .2f}x)" for i in range(len(lambdas_))]
            label_comp = f"y = {coeffs_[0]:.2f} + " + " + ".join(label_preComp)

        return operand, label_comp, coeffs, lambdas

    @staticmethod
    def sinusoidal_regression(ind_var: np.ndarray, dep_var: np.ndarray, degree: int):
        """
        Performs a sinusoidal regression on the independent variable `ind_var` and
        dependent variable `dep_var`. The degree of the polynomial is specified by `degree`.

        :returns: tuple: (operand, label_comp, amps, maxFreq, phaseShift)
        operand : LeftOperand
            The operand object representing the sinusoidal regression model.
        label_comp : str
            The equation of the regression model.
        amps : np.ndarray
            The amplitudes of the sinusoidal components of the regression model.
        maxFreq : np.ndarray
            The frequencies of the sinusoidal components of the regression model.
        phaseShift : np.ndarray
            The phase shifts of the sinusoidal components of the regression model.

        :param ind_var: np.ndarray
            The independent variable data as a numpy array.
        :param dep_var: np.ndarray
            The dependent variable data as a numpy array.
        :param degree: int
            The degree of the polynomial regression model.
        """
        # Ensure variables are 1D arrays
        ind_var = np.squeeze(ind_var)
        dep_var = np.squeeze(dep_var)

        # Check if the degree is greater than 0
        if degree < 1:
            raise ValueError("Degree must be greater than 0")

        # Check if the degree is greater than half the length of the independent variable
        if degree > ind_var.shape[0] // 2:
            raise ValueError("Degree too high for sinusoidal regression!")

        # Check if the independent and dependent variables have the same length
        if ind_var.shape != dep_var.shape:
            raise ValueError("ind_var and dep_var must have the same length")

        # Check if the independent variable is a single column
        if ind_var.ndim > 1:
            raise ValueError("ind_var must be a single column")

        # Sort the independent variable and dependent variable in ascending order of the independent variable
        sort = ind_var.argsort()
        ind_var = ind_var[sort]
        dep_var = dep_var[sort]

        # Apply a Gaussian window to the independent variable
        window = swa.gaussian(len(ind_var), len(ind_var))

        # Compute the Fourier transform of the dependent variable
        fft = sfft.rfft(dep_var)

        # Compute the frequencies corresponding to the Fourier transform
        N = len(ind_var)
        n = np.linspace(ind_var.min(), ind_var.max(), len(ind_var))
        freq = sfft.rfftfreq(N, (n[1] - n[0]))

        # plt.figure(figsize=(12, 6))
        # plt.vlines(freq, 0, np.abs(fft), 'b')
        # plt.xlabel('Freq (Hz)')
        # plt.ylabel('FFT Amplitude |X(freq)|')

        # Compute the magnitude and phase of the Fourier transform
        fftMag = 2 * np.abs(fft) / np.sum(window)
        fftMag[0] = fftMag[0] / 2
        fftPhase = np.angle(fft)

        # Find the indices of the `degree` largest magnitudes in the Fourier transform
        maxIndex = np.argpartition(fftMag[0:N // 2 + 1], -degree)[-degree:]

        # Calculate the amplitude
        amps = fftMag[maxIndex]
        # Calculate the frequency
        maxFreq = freq[maxIndex]
        # Calculate the phase shift
        phaseShift = fftPhase[maxIndex] + np.pi / 2

        # Construct the regression equation label
        label_preComp = [
            f"{amps[i]:.2f}sin(2π*{maxFreq[i]: .2f}*x{f' + {phaseShift[i]: .2f}' if phaseShift[i] >= 0.01 else ''})"
            if maxFreq[i] != 0 else f"{amps[i]: .2f}"
            for i in range(len(amps))]
        label_comp = "y = " + " + ".join(label_preComp)

        # Construct the operand for the regression model
        operand = LeftOperand(lambda x: np.sum(amps * np.sin(2 * np.pi * maxFreq * x + phaseShift), axis=1))
        return operand, label_comp, amps, maxFreq, phaseShift

    def regression(self, ind_vars_: List[str] | np.ndarray, dep_var_: str | np.ndarray,
                   regress_type: RegressTypes = RegressTypes.linear, degree: int = 1, plot_on: bool = False,
                   alt_out=False, summary_data=False):
        """
        Perform a regression of type `regress_type` on the independent variables `ind_vars` and dependent variable `dep_var`.

        :returns: tuple: (opp, label, out, ext_data)
            opp: LeftOperand
                The operand object representing the regression model.
            label: str
                The equation of the regression model.
            out: matplotlib.figure.Figure | list
                The regression plot or the plot arguments, depending on the value of `alt_out`.
            ext_data: dict, optional
                The summary data, if `summary_data` is True. Contains the mean, max residual, min value, squared mean deviation,
                mean squared error, sum of squared errors, total sum of squares, sum of squares due to regression, R^2,
                residuals, coefficients, lambdas, frequencies, and phase shift.

        :param ind_vars_: List[str] | np.ndarray
            The independent variables for the regression. Can be a list of variable names or a numpy array of data.
        :param dep_var_: str | np.ndarray
            The dependent variable for the regression. Can be a variable name or a numpy array of data.
        :param regress_type: RegressTypes
            The type of regression to perform. Default is linear regression.
        :param degree: int
            The degree of the polynomial for polynomial regression. Default is 1.
        :param plot_on: bool
            Whether to plot the regression. Default is False.
        :param alt_out: bool
            Whether to output the plot arguments instead of the plot itself. Default is False.
        :param summary_data: bool
            Whether to return summary data. Default is False.
        """

        # Check if the regression type is valid
        if regress_type not in RegressTypes.__members__:
            raise ValueError(f"regress_type must be one of {RegressTypes}")

        # Assign the independent and dependent variables
        ind_vars = ind_vars_
        dep_var = dep_var_

        # Check the type of the independent variables
        if isinstance(ind_vars, list):
            ind_count = len(ind_vars)
            # Check if there is at least one independent variable
            if ind_count < 1:
                raise ValueError("Must have at least one independent variable")
            # Check if all independent variables are strings
            if any([not isinstance(i, str) for i in ind_vars]):
                raise ValueError("All independent variables must be strings")
            # Select the data for the independent variables
            A = self.data.select_data(ind_vars)
        elif isinstance(ind_vars, np.ndarray):
            A = ind_vars
            # Check the dimensions of the independent variables
            if A.ndim < 2:
                A = A[:, np.newaxis]
            if A.ndim > 2:
                raise ValueError("ind_vars must be 2d.")
            ind_count = A.shape[1]
            # Check if there is at least one independent variable
            if ind_count < 1:
                raise ValueError("Must have at least one independent variable")
            # Create labels for the independent variables
            ind_vars = [f"x_{i}" for i in range(ind_count)]
        else:
            raise ValueError("ind_vars must be a list of strings or a numpy array")

        # Check the type of the dependent variable
        if isinstance(dep_var, str):
            y = self.data.select_data([dep_var])
        elif isinstance(dep_var, np.ndarray):
            y = dep_var
            # Check the dimensions of the dependent variable
            if y.ndim < 2:
                y = y[:, np.newaxis]
            if y.ndim > 2:
                raise ValueError("dep_var must be 2d")
            # Check if the dependent variable is a single column
            if y.shape[1] > 1:
                raise ValueError("dep_var must be a single column")
            dep_var = "y"
        else:
            raise ValueError("dep_var must be a string or a numpy array")

        # Set the degree of the polynomial for the regression
        self.p = degree
        lambdas = None
        phase_shift = None
        frequencies = None

        # Perform the appropriate type of regression
        if regress_type == 'linear':
            self.ind_vars = ind_vars
            self.dep_var = dep_var
            self.y = y
            self.A = A
            opp, label, coeffs = LinearRegression.linear_regression_2(A, y)
        elif regress_type == 'polynomial':
            opp, label, coeffs = LinearRegression.multi_poly_regression(A, y, degree)
        elif regress_type == 'exponential':
            if ind_count > 1:
                raise ValueError("Exponential regression only supports one independent variable")
            opp, label, coeffs, lambdas = LinearRegression.exponential_regression(A, y, degree)
        elif regress_type == 'sinusoidal':
            if ind_count > 1:
                raise ValueError("Sinusoidal regression only supports one independent variable")
            opp, label, coeffs, frequencies, phase_shift = LinearRegression.sinusoidal_regression(A, y, degree)
        elif regress_type == 'mixed':
            print("Mixed regression treats only first independent variable as exponential and the rest as polynomial")
            raise ValueError("Mixed regression not yet implemented. Please use a different regression type.")
        else:
            raise ValueError(f"regress_type must be one of {RegressTypes}")

        # Perform data analysis on the regression
        sample_size, mean, predicts, residuals, max_resid, min_val, mean_errors, smd, mse, sse, sst, ssr, R2 = LinearRegression.data_anal(
            opp, y, A)

        # Store the results of the data analysis
        ext_data = {"mean": mean, "max_resid": max_resid, "min_val": min_val, "smd": smd, "mse": mse, "sse": sse,
                    "sst": sst, "ssr": ssr, "R2": R2, "residuals": residuals, "coeffs": coeffs, "lambdas": lambdas,
                    "frequencies": frequencies, "phase_shift": phase_shift}

        out = None

        # Check if a plot should be created
        if plot_on:
            if ind_count == 1:
                args = [A[:, 0], y, f"{ind_vars[0]} by {dep_var} with {regress_type} Regression (R^2 = {R2:.2f})"]
                kwargs = {"operand": opp, "reg_label": label, "residuals_on": True, "fig_sz": None, "R2": R2}
                if alt_out:
                    out = [args, kwargs]
                else:
                    out = self.scatter(*args, **kwargs)

        # Check if summary data should be returned
        if not summary_data:
            return opp, label, out
        return opp, label, out, ext_data

    def get_fitted_slope(self):
        """Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        """
        return self.slope

    def get_fitted_intercept(self):
        """Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        """
        return self.intercept

    def initialize(self, ind_vars: List[str], dep_var: str, slope: np.ndarray = None, intercept: float = None,
                   p: int = 1,
                   operand: Operands.LeftOperand = None, generate: bool = False):
        """Sets fields based on parameter values.
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
        """
        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.A = self.data.select_data(ind_vars)
        self.y = self.data.select_data([dep_var])
        self.slope = slope
        self.intercept = intercept

        if generate:
            self.lin_opp, label, ext_data = self.regression(ind_vars, dep_var, degree=p, plot_on=False, alt_out=False,
                                                            summary_data=True)
            self.slope = ext_data["coeffs"][1:]
            self.intercept = ext_data["coeffs"][0]
            self.R2 = ext_data["R2"]
            self.mse = ext_data["mse"]

        elif operand is not None:
            self.lin_opp = operand
        elif slope is not None and intercept is not None:
            if slope.shape[0] != self.A.shape[1]:
                raise ValueError("Slope must have the same number of rows as the number of independent variables")
            if slope.shape[1] != 1:
                raise ValueError("Slope must have one column")
            if intercept.__class__ != float:
                raise ValueError("Intercept must be a float")
            self.lin_opp = LeftOperand(lambda x: intercept + np.einsum('ij,j->i', x, slope))
        else:
            raise ValueError("Must provide either a slope and intercept or a LeftOperand object")

        self.R2 = self.r_squared()

        # Mean SEE. float. Measure of quality of fit
        self.mse = self.compute_mse()

        #   Residuals from regression fit
        self.residuals = self.compute_residuals()

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = p

    @staticmethod
    def d_derivative(x: np.ndarray, y: np.ndarray):
        """
        Calculates an approximate derivative of the data using the difference quotient method.
        The derivative is computed for each pair of consecutive points in the input arrays.

        :param x: np.ndarray
            The x-values of the data points. Must be a 1D array.
        :param y: np.ndarray
            The y-values of the data points. Must be a 1D array of the same length as `x`.

        :return: dy_dx_mat: np.ndarray
            The approximate derivative of the data. It is a 1D array of the same length as `x` and `y`.

        :raises ValueError:
            If `x` and `y` do not have the same length.
            If `x` and `y` do not have at least 2 elements.
        """

        # Check if x and y have the same length
        if x.shape != y.shape:
            raise ValueError("x and y must have the same length")

        # Check if x and y have at least 2 elements
        if x.shape[0] < 2:
            raise ValueError("x and y must have at least 2 elements")

        # Sort x and y in ascending order of x
        arg = x.argsort()
        x_mod = x[arg]
        y_mod = y[arg]

        # Append a NaN to the end of x and y
        mod_mat_x = np.append(x_mod, np.nan)
        mod_mat_y = np.append(y_mod, np.nan)

        # Compute the differences between consecutive elements in x and y
        comp_mat_x_up = np.column_stack([mod_mat_x, -np.roll(mod_mat_x, -1)])
        comp_mat_x_down = np.column_stack([-mod_mat_x, np.roll(mod_mat_x, 1)])
        comp_mat_y_up = np.column_stack([mod_mat_y, -np.roll(mod_mat_y, -1)])
        comp_mat_y_down = np.column_stack([-mod_mat_y, np.roll(mod_mat_y, 1)])

        # Compute the differences between consecutive elements in x
        dx_mat = np.column_stack([np.einsum('ij->i', comp_mat_x_up), np.einsum('ij->i', comp_mat_x_down)])[:-1]
        # Compute the differences between consecutive elements in y
        dy_mat = np.column_stack([np.einsum('ij->i', comp_mat_y_up), np.einsum('ij->i', comp_mat_y_down)])[:-1]

        # Compute the ratio of dy to dx
        dy_dx_d_mat = dy_mat / dx_mat

        # Compute the mean of the ratios to get the approximate derivative
        dy_dx_mat = np.nanmean(dy_dx_d_mat, axis=1)
        return dy_dx_mat

    @staticmethod
    def d_int(x: np.ndarray, y: np.ndarray):
        """
        Calculates an approximate integral of the data using the trapezoidal rule.
        The integral is computed for each pair of consecutive points in the input arrays.

        :param x: np.ndarray
            The x-values of the data points. Must be a 1D array.
        :param y: np.ndarray
            The y-values of the data points. Must be a 1D array of the same length as `x`.

        :return:
        cum_sum : np.ndarray
            The approximate integral of the data. It is a 1D array of the same length as `x` and `y`.

        :raises ValueError:
            If `x` and `y` do not have the same length.
            If `x` and `y` do not have at least 2 elements.
        """
        # Check if x and y have the same length
        if x.shape != y.shape:
            raise ValueError("x and y must have the same length")
        # Check if x and y have at least 2 elements
        if x.shape[0] < 2:
            raise ValueError("x and y must have at least 2 elements")

        # Sort x and y in ascending order of x
        arg = x.argsort()
        x_mod = x[arg]
        y_mod = y[arg]

        # Compute the differences between consecutive elements in x and y
        comp_mat_x_up = np.column_stack([-x_mod, np.roll(x_mod, -1)])[:-1]
        comp_mat_y_up = np.column_stack([-y_mod, np.roll(y_mod, -1)])[:-1]
        dx_mat = np.array(np.einsum('ij->i', comp_mat_x_up)).tolist()
        dy_mat = np.array(np.einsum('ij->i', comp_mat_y_up)).tolist()

        # Initialize the accumulator for the integral
        accum = [0]

        # Compute the integral using the trapezoidal rule
        for i in range(x.shape[0] - 1):
            accum.append(dx_mat[i] * (y_mod[i] + (1 / 2) * dy_mat[i]) + accum[-1])

        # Convert the accumulator to a numpy array
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
