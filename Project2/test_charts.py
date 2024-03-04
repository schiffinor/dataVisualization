from charts import *

# Test case 1
values = np.array([[1, 2, 3], [4, 5, 6]])
header1_labels = ['2020', '2021']
header2_levels = ['Red', 'Green', 'Blue']
title = 'Test Plot 1'
grouped_sidebarplot(values, header1_labels, header2_levels, title)

# Test case 2
values = np.array([[10, 20, 30], [40, 50, 60]])
header1_labels = ['A', 'B']
header2_levels = ['X', 'Y', 'Z']
title = 'Test Plot 2'
grouped_sidebarplot(values, header1_labels, header2_levels, title)

# Test case 3
values = np.array([[100, 200, 300], [400, 500, 600]])
header1_labels = ['Group 1', 'Group 2']
header2_levels = ['Category 1', 'Category 2', 'Category 3']
title = 'Test Plot 3'
grouped_sidebarplot(values, header1_labels, header2_levels, title)

# Test case 4
