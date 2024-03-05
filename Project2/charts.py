"""charts.py
Plotting functions for categorical data
YOUR NAME HERE
CS 251: Data Analysis and Visualization
Spring 2024
"""
import numpy as np
import matplotlib.pyplot as plt


def sidebarplot(values, labels, title, show_counts=True, figsize=(6, 7), sort_by='na', format_as="d"):
    """Horizontal bar plot with bar lengths `values` (the "x values") with associated labels `labels` on the y axis.

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
    """
    formatter = {'value': '%.2f', 'percent': '%.2f%%', 'd': '%d'}[format_as]
    plt.figure(figsize=figsize)
    values, labels = sort(values, labels, sort_by)
    total = np.sum(values)
    if format_as == "percent":
        values = values / total * 100
    ax = plt.barh(np.arange(len(values)), values, align='center', color='#ffffff', edgecolor='#000000')
    plt.yticks(np.arange(len(values)), labels)
    plt.title(title)
    splitter = title.split(' by ')
    if len(splitter) > 1:
        plt.xlabel(splitter[0], rotation=0, labelpad=20)
        plt.ylabel(splitter[1], rotation=0, labelpad=20)
    if show_counts:
        plt.bar_label(ax, fmt=formatter, padding=3)
    # plt.tight_layout()
    plt.autoscale()



def sort(values, labels, sort_by='na'):
    """Sort the arrays `values` and `labels` in the same way so that corresponding items in either array stay matched up
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
    """
    args = np.argsort({'na': range(len(values)), 'value': values, 'label': labels}[sort_by])
    return values[args], labels[args]


def grouped_sidebarplot(values: np.ndarray, header1_labels, header2_levels, title, show_counts=True, figsize=(6, 7), format_as="d"):
    """
    Horizontal side-by-side bar plot with bar lengths `values` (the "x values") with associated labels.
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
    """
    formatter = {'value': '%.2f', 'percent': '%.2f%%', 'd': '%d'}[format_as]
    plt.figure(figsize=figsize)
    num_groups = len(header1_labels)
    num_bars = len(header2_levels)
    bar_width = 0.8 / num_bars
    values = values
    sums_main = np.sum(values, axis=1)
    sub_sums = np.sum(values, axis=0)
    header1_labels = sort(header1_labels, sums_main, 'label')[0]
    header2_levels = sort(header2_levels, sub_sums, 'label')[0]
    values = sort(values, sums_main, 'label')[0]
    for i in range(num_groups):
        value_row = values[i]
        value_row = sort(value_row, sub_sums, 'label')[0]
        values[i] = value_row
    plots = []
    total = np.sum(values, axis=0)
    for i in range(num_bars):
        value_set = values[:, i]
        if format_as == "percent":
            value_set = value_set / total[i] * 100
        ys = np.arange(num_groups) + i * bar_width
        sub = plt.barh(ys, value_set, bar_width, align='center', label=header2_levels[i])
        plots.append(sub)
        if show_counts:
            plt.bar_label(sub, fmt=formatter, padding=3)
    plt.yticks(np.arange(num_groups) + bar_width * (num_bars - 1) / 2, header1_labels)
    plt.title(title)
    splitter = title.split(' by ')
    if len(splitter) > 1:
        plt.xlabel(splitter[0], labelpad=20)
        plt.ylabel(splitter[1], labelpad=20)
    plt.legend(title='Type', bbox_to_anchor=(1.05, 1.0), loc='upper left', reverse=True)
    plt.autoscale()

