'''
linear_regression.py
Subclass of Analysis that performs linear regression on data
Roman Schiffino
CS251 Data Analysis Visualization
Spring 2024
'''
import time

import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
from Operands import *
from typing import List, Dict
from regressTypes import *
from data import Data
import scipy.fft as sfft
import scipy.signal.windows as swa

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
        self.lin_opp = LeftOperand(lambda x: self.intercept + np.einsum('ij,ji->i', x, self.slope)[:, np.newaxis])
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
        augA = np.hstack([np.ones((self.A.shape[0], 1)), self.A])
        c, residue, rank, s = sla.lstsq(augA, self.y)
        self.slope = c[1:]
        self.intercept = c[0][0]
        predicts = self.predict()
        self.residuals = self.compute_residuals(predicts)
        self.R2 = self.r_squared(predicts)


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
        return self.lin_opp | X

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
        return 1/len(self.y) * np.einsum('i,i->', residuals.flatten(), residuals.flatten())

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
        mean_errors = self.compute_residuals(1/len(self.y) * np.einsum('i->', self.y.flatten()))
        return 1/len(self.y) * np.einsum('i,i->', mean_errors.flatten(), mean_errors.flatten())

    def scatter(self, ind_var, dep_var, title, residuals_on=False, fig_sz=(12, 10), colors=None, regression_line=True, regress_type='linear', poly_deg=1):
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
        if regress_type not in RegressTypes:
            raise ValueError(f"regress_type must be one of {RegressTypes}")
        if regress_type == '':
            pass
        if colors is None:
            colors = ['blue', 'red']
        x_data = self.data.select_data([ind_var])
        y_data = self.data.select_data([dep_var])
        plt.scatter(x_data, y_data)
        x_max = np.max(x_data)
        x_min = np.min(x_data)
        y_max = np.max(y_data)
        y_min = np.min(y_data)
        d_domain = x_max - x_min
        d_range = y_max - y_min
        line_x = np.linspace(x_min - 0.1 * d_domain, x_max + 0.1 * d_domain, 100)
        line_y = self.lin_opp | line_x
        all_coeffs = [self.intercept] + [self.slope[0]]
        # label_preComp = [f"{}" for i in range(len(amps))]
        # label_comp = "y = " + " + ".join(label_preComp)
        # plt.plot(line_x, line_y, label=f"y = {c[0]:.2f}x + {c[1]:.2f}")
        # plt.title(f"Scatterplot of X and Y with Linear Regression (R^2 = {r2:.2f})")
        plt.xlabel("X")
        plt.ylabel("Y", rotation=0, labelpad=10)
        plt.legend()
        plt.show()
        plt.figure(figsize=fig_sz)
        # plt.plot(line_x, line_y, "r", label=label_comp)
        plt.scatter(x_data, y_data)
        # plt.vlines(x_data[sort], -20, -20 + (y_data[sort] - line_y_), color='purple', alpha=0.5)
        plt.xlabel("X")
        plt.ylabel("Y", rotation=0, labelpad=10)
        plt.legend()
        plt.show()

    def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
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
        pass

    def make_polynomial_matrix(self, A, degree):
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

    def comp_matrix(self, A, degree):
        A_transpose = A.T
        mat_List = []
        index = 0
        for row in A_transpose.tolist():
            row_ = np.array(row)
            temp_augA = self.make_polynomial_matrix(row_, degree)
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
        self.multi_poly_regression(A_data, y_data, degree)

    def multi_poly_regression(self, ind_vars: np.ndarray, dep_var: np.ndarray, degree: int):
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
        augA = self.comp_matrix(ind_vars, degree)
        print(f"augA: {augA}")
        print(f"augA.shape: {augA.shape}")
        c, residue, rank, s = sla.lstsq(augA, dep_var)
        intercept = c[0][0]
        poly_coeffs = c[1:]
        var_poly_coeffs = poly_coeffs.reshape((degree, ind_vars.shape[1])).T
        print(f"var_poly_coeffs: {var_poly_coeffs}")
        var_poly_coeffs = var_poly_coeffs.reshape((ind_vars.shape[1], 1, degree))
        print(f"var_poly_coeffs: {var_poly_coeffs.shape}")
        print(f"var_poly_coeffs: {var_poly_coeffs}")
        operand = LeftOperand(lambda x: intercept + np.einsum('ijk,ijl->j', np.tile(self.comp_matrix(x, degree)[np.newaxis, :, 1:], (2, 1, 1)), var_poly_coeffs)[:, np.newaxis])
        return operand

    def exponential_regression(self, ind_vars: np.ndarray, dep_var: np.ndarray):
        pass

    def sinusoidal_regression(self, ind_var: np.ndarray, dep_var: np.ndarray, degree: int):
        if degree < 1:
            raise ValueError("Degree must be greater than 0")
        if degree > ind_var.shape[0] // 2:
            raise ValueError("Degree too high for sinusoidal regression!")
        if ind_var.shape != dep_var.shape:
            raise ValueError("ind_var and dep_var must have the same length")
        if ind_var.ndim > 1:
            raise ValueError("ind_var must be a single column")

        sort = ind_var.flatten().argsort()
        ind_var = ind_var[sort]
        dep_var = dep_var[sort]
        window = swa.gaussian(len(ind_var), len(ind_var))
        fft = sfft.rfft(dep_var)

        N = len(ind_var)
        n = np.linspace(ind_var.min(), ind_var.max(), len(ind_var))
        # get the sampling rate
        sr = 1. / (n[1] - n[0])
        T = ind_var.max() - ind_var.min()
        freq = sfft.rfftfreq(N, (n[1] - n[0]))

        print(f"fft: {fft}")
        print(f"n: {n}")
        print(f"freq: {freq}")
        print(f"sr: {sr}")
        print(f"T: {T}")

        plt.figure(figsize=(12, 6))
        plt.vlines(freq, 0, np.abs(fft), 'b')
        plt.xlabel('Freq (Hz)')
        plt.ylabel('FFT Amplitude |X(freq)|')
        plt.show()

        fftMag = 2 * np.abs(fft) / np.sum(window)
        fftMag[0] = fftMag[0] / 2
        fftPhase = np.angle(fft)
        print(f"fftMag: {fftMag}")
        print(f"fftPhase: {fftPhase}")

        # Index of max
        maxIndex = np.argpartition(fftMag[0:N // 2 + 1], -degree)[-degree:]
        print(f"maxIndex: {maxIndex}")

        # Calculate the amplitude
        amps = fftMag[maxIndex]
        # Calculate the frequency
        maxFreq = freq[maxIndex]
        '''if 0 in maxFreq:
            raise ValueError("Degree too high for sinusoidal regression!\n 0 frequency found in maxFreq.")
        '''# Calculate the phase shift
        phaseShift = fftPhase[maxIndex] + np.pi / 2

        print(f"Amplitude: {amps}")
        print(f"Frequency: {maxFreq}")
        print(f"Phase: {phaseShift}")
        x_data_ = ind_var[sort][:, np.newaxis]
        line_y_ = np.sum(amps * np.sin(2 * np.pi * maxFreq * x_data_ + phaseShift), axis=1)
        label_preComp = [
            f"{amps[i]:.2f}sin(2π*{maxFreq[i]: .2f}*x + {phaseShift[i]: .2f})"
            for i in range(len(amps))]
        label_comp = "y = " + " + ".join(label_preComp)
        print(dep_var.shape)
        print(line_y_.shape)
        residue = np.einsum('i,i->', dep_var[sort] - line_y_, dep_var[sort] - line_y_)
        print(f"Residue: {residue}")
        mean = np.mean(dep_var[sort])
        smd = np.einsum('i,i->', dep_var[sort] - mean, dep_var[sort] - mean)
        r2 = 1 - (residue / smd)
        print(f"smd: {smd}")
        print(f"r2: {r2}")
        operand = LeftOperand(lambda x: np.sum(amps * np.sin(2 * np.pi * maxFreq * x + phaseShift), axis=1))
        print(line_y_.shape)
        residue = np.einsum('i,i->', dep_var[sort] - line_y_, dep_var[sort] - line_y_)
        print(f"Residue: {residue}")
        r2 = 1 - (residue / smd)
        print(f"smd: {smd}")
        print(f"r2: {r2}")
        plt.figure(figsize=(20, 10))
        plt.plot(n, operand | (n[:, np.newaxis]), "r", label=label_comp)
        plt.scatter(ind_var, dep_var)
        plt.vlines(ind_var[sort], -20, -20 + (dep_var[sort] - line_y_), color='purple', alpha=0.5)
        plt.xlabel("X")
        plt.ylabel("Y", rotation=0, labelpad=10)
        plt.legend()
        plt.show()
        return operand, label_comp

    def regression(self, ind_vars_: List[str], dep_var_: str, regress_type: str = 'linear', degree: int = 1):
        '''Perform regression — generalizes self.linear_regression to polynomial curves
        (Week 2)

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and an added column of 1s for the intercept.
        '''
        if regress_type not in RegressTypes:
            raise ValueError(f"regress_type must be one of {RegressTypes}")
        ind_vars = ind_vars_
        dep_var = dep_var_
        ind_count = len(ind_vars)
        if ind_count < 1:
            raise ValueError("Must have at least one independent variable")
        self.p = degree
        y = self.data.select_data([dep_var])
        A = self.data.select_data(ind_vars)
        if regress_type == 'linear':
            self.ind_vars = ind_vars
            self.dep_var = dep_var
            self.y = y
            self.A = A
            self.linear_regression(ind_vars, dep_var)
            slope = self.slope
            intercept = self.intercept
            predicts = self.predict()
            residuals = self.residuals
            R2 = self.R2
        elif regress_type == 'polynomial':
            operand = self.multi_poly_regression(A, y, degree)
        elif regress_type == 'exponential':
            if ind_count > 1:
                raise ValueError("Exponential regression only supports one independent variable")
            slope, intercept, predicts, residuals, R2 = self.exponential_regression(A, y)
        elif regress_type == 'sinusoidal':
            if ind_count > 1:
                raise ValueError("Sinusoidal regression only supports one independent variable")
            operand = self.sinusoidal_regression(A, y, degree)
        elif regress_type == 'mixed':
            print("Mixed regression treats only first independent variable as exponential and the rest as polynomial")
            slope, intercept, predicts, residuals, R2 = self.mixed_regression(A, y, degree)
        else:
            raise ValueError(f"regress_type must be one of {RegressTypes}")




    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        pass

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        pass

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
        pass

    def d_derivative(self, x: np.ndarray, y: np.ndarray):
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


if __name__ == '__main__':
    data = Data('data/iris.csv')
    analysis = LinearRegression(data)
    # Create a simple linear function y = 2x^2 + 3
    x_ = np.arange(0, 10, 0.1)
    y_ = 2 * x_ * x_ + 3

    # Calculate derivative
    derivative = analysis.d_derivative(x_, y_)

    # Since y = 2x^2 + 3 is a linear function, its derivative is 4x
    # Let's compare the calculated derivative with the expected derivative
    expected_derivative = 4 * x_
    print("Calculated derivative: ", derivative)
    print("Expected derivative: ", expected_derivative)
    np.testing.assert_allclose(derivative[1:-1], expected_derivative[1:-1], rtol=1e-4)

    # Perform sinusoidal regression
    data = np.genfromtxt("Lab03a/data/mystery_data_1.csv", delimiter=",")[2:]
    print(data[:10])
    print(data.shape)

    x_dat = data[:, 0]
    y_dat = data[:, 1]
    operand, label = analysis.sinusoidal_regression(x_dat, y_dat, 1)

    x_m = np.linspace(0, 4 * np.pi, 5000)
    y_m = 2 * np.sin(x_m) + 3 * np.sin(2 * x_m) + 4 * np.sin(3 * x_m) + 5 + np.random.uniform(-0.5, 0.5, 5000)
    print(f"x_m: {x_m}")
    print(f"y_m: {y_m}")
    operand, label = analysis.sinusoidal_regression(x_m, y_m, 4)
    print("Label: ", label)
    print(f"Operand: {operand | x_m[:, np.newaxis]}")
    plt.plot(x_m, operand | x_m[:, np.newaxis], "r", label=label)
    plt.scatter(x_m, y_m)
    plt.xlabel("X")
    plt.ylabel("Y", rotation=0, labelpad=10)
    plt.legend()
    plt.show()
    print(f"Operand: {operand | np.array([0,1,2,3,4])[:, np.newaxis]}")
    # Get fitted slope and intercept
    x_m = np.linspace(-10, 10, 5000)
    z_m = np.linspace(-10, 10, 5000)
    y_m = x_m**2 + np.random.uniform(-0.5, 0.5, 5000)
    operand = analysis.multi_poly_regression(x_m[:, np.newaxis], y_m[:, np.newaxis], 2)
    print("Label: ", label)
    print(f"Operand: {operand | x_m[:, np.newaxis]}")
    print(operand | x_m[:, np.newaxis])
    plt.plot(x_m, operand | x_m[:, np.newaxis], "r", label=label)
    plt.scatter(x_m, y_m)
    plt.legend()
    plt.show()
    print(f"Operand: {operand | np.array([0, 1, 2, 3, 4])[:, np.newaxis]}")
    # Get fitted slope and intercept
    print("All tests passed!")
