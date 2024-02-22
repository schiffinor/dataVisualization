'''charts.py
Plotting functions for categorical data
YOUR NAME HERE
CS 251: Data Analysis and Visualization
Spring 2024
'''
import numpy as np
import matplotlib.pyplot as plt


def sidebarplot(values, labels, title, show_counts=True, figsize=(6, 7)):
    '''Horizontal bar plot with bar lengths `values` (the "x values") with associated labels `labels` on the y axis.

    Parameters:
    -----------
    values: ndarray. shape=(num_levels). Each numeric value to plot as a separate horizontal bar coming out of the y axis.
    labels: list of str. len=num_labels. Labels associated with each bar / numeric value in `values`.
    title: str. Title of the plot.
    show_counts: bool. Whether to show each numeric value as text next to each bar.
    fig_sz: tuple of 2 ints. The width and height of the figure.

    NOTE:
    - Assign the output of plt.barh to a variable named ax. i.e.
        ax = plt.barh(...)
    If show_counts is set to True, then add the code:
        if show_counts:
            plt.bar_label(ax, values)    
    to make the values appear on the plot as text next to each bar.
    - If your values show up next to the bars with many significant digits, add the following line of code to have Numpy
    round each displayed value to the nearest 0.01:
        values = np.round(values, 2)
    '''
    plt.figure(figsize=figsize)
    values = np.round(values, 2)
    ax = plt.barh(labels, values, align='center', color='#ffffff', edgecolor='#000000')
    plt.title(title)
    splitter = title.split(' by ')
    plt.xlabel(splitter[0])
    plt.ylabel(splitter[1])
    if show_counts:
        plt.bar_label(ax, values)



def sort(values, labels, sort_by='na'):
    '''Sort the arrays `values` and `labels` in the same way so that corresponding items in either array stay matched up
    after the sort.

    Parameters:
    -----------
    values: ndarray. shape=(num_levels,). One array that should be sorted
    labels: ndarray. shape=(num_levels,). Other array that should be sorted
    sort_by: str. Method by which the arrays should be sorted. There are 3 possible values:
        1. 'na': Keep the arrays as-is, no sorting takes place.
        2. 'value': Reorder both arrays such that the array `values` is sorted in ascending order.
        3. 'label': Reorder both arrays such that the array `labels` is sorted in ascending order.

    Returns:
    -----------
    ndarray. shape=(num_levels,). Sorted `values` array. Corresponding values in `labels` remain matched up.
    ndarray. shape=(num_levels,). Sorted `labels` array. Corresponding values in `values` remain matched up.


    NOTE:
    - np.argsort might be helpful here.
    '''
    mapping = {'na': range(len(values)), 'value': values, 'label': labels}
    args = np.argsort(mapping[sort_by])
    return values[args], labels[args]


def grouped_sidebarplot(values, header1_labels, header2_levels, title, figsize=(6, 7)):
    '''Horizontal side-by-side bar plot with bar lengths `values` (the "x values") with associated labels.
    `header1_labels` are the levels of `header`, which appear on the y axis. Each level applies to ONE group of bars next
    to each other. `header2_labels` are the levels that appear in the legend and correspond to different color bars.

    POSSIBLE EXTENTION. NOT REQUIRED FOR BASE PROJECT

    (Useful for plotting numeric values associated with combinations of two categorical variables header1 and header2)

    Parameters:
    -----------
    values: ndarray. shape=(num_levels). Each numeric value to plot as a separate horizontal bar coming out of the y axis.
    labels: list of str. len=num_labels. Labels associated with each bar / numeric value in `values`.
    title: str. Title of the plot.
    show_counts: bool. Whether to show each numeric value as text next to each bar.
    fig_sz: tuple of 2 ints. The width and height of the figure.

    Example:
    -----------
    header1_labels = ['2020', '2021']
    header2_labels = ['Red', 'Green', 'Blue']
    The side-by-side bar plot looks like:

    '2021' 'Red'  ----------
     y=2   'Green'------
           'Blue' ----------------

    '2020' 'Red'  -------------------
     y=1   'Green'-------------------
           'Blue' ---------

    In the above example, the colors also describe the actual colors of the bars.

    NOTE:
    - You can use plt.barh, but there are offset to compute between the bars...
    '''
    pass
