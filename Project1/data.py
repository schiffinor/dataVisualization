"""
data.py
Reads CSV files, stores data, access/filter data by variable name
Roman Schiffino
CS 251: Data Analysis and Visualization
Spring 2024
"""

import math
from typing import List, Dict

import dateutil.parser as d_parse
import numpy as np

import matrix as m
from dataTypes import DataTypes as dT
from dataTypesTrim import DataTypes as dTT


class Data:
    """
    Represents data read in from .csv files
    """

    def __init__(self, filepath=None, headers=None, data=None, header2col=None, cats2levels=None, allDataTypes=False,
                 compatMode=False):
        """
        Data object constructor

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file
        headers: Python list of strings or None. List of strings that explain the name of each column of data.
        data: ndarray or None. shape=(N, M).
            N is the number of data samples (rows) in the dataset and M is the number of variables (cols) in the dataset.
            2D numpy array of the dataset’s values, all formatted as floats.
            NOTE: In Week 1, don't worry working with ndarrays yet. Assume it will be passed in as None for now.
        header2col: Python dictionary or None.
                Maps header (var str name) to column index (int).
                Example: "sepal_length" -> 0
        cats2levels: Python dictionary or None.
                Maps each categorical variable header (var str name) to a list of the unique levels (strings)
                Example:

                For a CSV file that looks like:

                letter,number,greeting
                categorical,categorical,categorical
                a,1,hi
                b,2,hi
                c,2,hi

                cats2levels looks like (key -> value):
                'letter' -> ['a', 'b', 'c']
                'number' -> ['1', '2']
                'greeting' -> ['hi']
        """
        # For utility and organization.
        self.file = None
        self.filepath = None
        self.headers = headers
        self.whole_headers = None
        self.var_data_type = None
        self.data_array = None
        self.whole_data_array = None
        self.whole_data_list = None
        self.data = data
        self.header2col = header2col
        self.whole_header2col = None
        # I love ternary operators if that wasn't immediately obvious. (Java's is better.)
        self.col2header = None if header2col is None else dict(zip(header2col.values(), header2col.keys()))
        self.whole_col2header = None
        # Basically I assume that pythons dict is a hashmap. Hashmap is more efficient than a list.
        self.cats2levels = {} if cats2levels is None else cats2levels
        self.cats2level_dicts = {} if cats2levels is None else None
        self.levels2cats = {} if cats2levels is None else cats2levels
        self.levels2cats_dicts = {} if cats2levels is None else None
        self.allDataTypes = allDataTypes
        # To meet the requirements of test files this was added. Basically restricts possible data types.
        self.dTRef = dT if allDataTypes else dTT
        # Also to meet the requirements of test files this was added.
        # My base code is quite robust and handles a lot of different stuff by default.
        # This is a feature to limit that handling. Causes error if all data types are missing.
        self.compatMode = compatMode

        if filepath is not None:
            self.read(filepath)

    def read(self, filepath):
        """
        Read in the .csv file `filepath` in 2D tabular format. Convert to numpy ndarray called `self.data` at the end
        (think of this as a 2D array or table).

        Format of `self.data`:
            Rows should correspond to i-th data sample.
            Cols should correspond to j-th variable / feature.

        Parameters:
        -----------
        filepath: str or None. Path to data .csv file

        Returns:
        -----------
        None. (No return value).
            NOTE: In the future, the Returns section will be omitted from docstrings if there should be nothing returned


        """
        if filepath is not None and len(filepath) != 0:
            if self.filepath != filepath:
                self.filepath = filepath
            else:
                print("Filepath is already set to {}".format(self.filepath))
                print("Re-reading data from {}".format(self.filepath))
            try:
                self.file = open(self.filepath, 'r')
                print("Reading data from file: {}".format(self.filepath))
            except OSError as e:
                print("Could not open file: {}".format(self.filepath))
                raise RuntimeError(e.strerror)
            raw_file_lines = self.file.readlines()
            file_lines = []
            for raw_line in raw_file_lines:
                line = raw_line.strip("\n")
                file_lines.append(line)
            self.file.close()

            raw_headers = file_lines[0].split(',')
            headers = []
            for raw_var_name in raw_headers:
                var_name = raw_var_name.strip()
                headers.append(var_name)
            self.whole_headers = headers
            # print("Headers: {}".format(self.headers))

            self.whole_header2col = dict(zip(self.whole_headers, range(len(self.whole_headers))))
            self.whole_col2header = dict(zip(range(len(self.whole_headers)), self.whole_headers))
            raw_data_types = file_lines[1].split(',')
            data_types = []
            for index, raw_data_type in enumerate(raw_data_types):
                data_type = raw_data_type.strip()
                # print("Data type: {}".format(data_type))
                if data_type not in self.dTRef.member_names_:
                    print("Invalid data type: {}\nIgnoring Column.\n".format(data_type))
                    data_types.append(self.dTRef["missing"])
                else:
                    data_types.append(self.dTRef[data_type])
            self.var_data_type = data_types
            if all(d_type.name == "missing" for d_type in self.var_data_type) and len(
                    self.var_data_type) > 0 and not self.compatMode:
                raise ValueError("All data types invalid.\n"
                                 "Input file likely missing data types.\n"
                                 "Second line must list data types."
                                 "Second line processed: {}".format(file_lines[1]))
            # print("Data types: {}".format(self.var_data_type))
            # print("Data types: {}".format(list(map(lambda var: var.name, self.var_data_type))))
            for index, datum in enumerate(self.var_data_type):
                if datum.name == "categorical":
                    self.cats2levels[self.whole_headers[index]] = []
                    self.cats2level_dicts[self.whole_headers[index]] = {}
                    self.levels2cats[self.whole_headers[index]] = []
                    self.levels2cats_dicts[self.whole_headers[index]] = {}
            self.whole_data_list = []
            for row_index, line in enumerate(file_lines[2:]):
                raw_data = line.split(',')
                data = []
                for index, raw_datum in enumerate(raw_data):
                    datum = raw_datum.strip() if self.var_data_type[index].name != "string" else raw_datum
                    if datum == "":
                        if self.var_data_type[index].name == "numeric":
                            data.append(np.nan)
                        elif self.var_data_type[index].name == "categorical":
                            category = "Missing"
                            temp_header = self.whole_col2header[index]
                            temp_list = self.cats2levels[temp_header]
                            temp_dict = self.cats2level_dicts[temp_header]
                            if category not in temp_list:
                                temp_list.append(category)
                                index = len(temp_list) - 1
                                self.cats2level_dicts[temp_header][temp_list[index]] = index
                                self.levels2cats_dicts[temp_header][index] = category
                                temp_dict = self.cats2level_dicts[temp_header]
                            data.append(temp_dict[category])
                        elif self.var_data_type[index].name == "string":
                            data.append(datum)
                        elif self.var_data_type[index].name == "date":
                            data.append("N/A")
                        elif self.var_data_type[index].name == "missing":
                            data.append(datum)
                        else:
                            raise ValueError("IMPOSSIBLE! Invalid data type: {}".format(self.var_data_type[index]))
                    else:
                        if self.var_data_type[index].name == "numeric":
                            try:
                                number = float(datum)
                            except OverflowError:
                                number = float("inf")
                                print("Overflow Error: {} is too large.".format(datum))
                            except ValueError:
                                number = np.nan
                                print("Data Error: {} is not a number.".format(datum))
                            data.append(number)
                        elif self.var_data_type[index].name == "categorical":
                            temp_header = self.whole_col2header[index]
                            temp_list = self.cats2levels[temp_header]
                            temp_dict = self.cats2level_dicts[temp_header]
                            if datum not in temp_list:
                                temp_list.append(datum)
                                index = len(temp_list) - 1
                                self.cats2level_dicts[temp_header][temp_list[index]] = index
                                self.levels2cats_dicts[temp_header][index] = datum
                                temp_dict = self.cats2level_dicts[temp_header]
                            data.append(temp_dict[datum])
                        elif self.var_data_type[index].name == "string":
                            data.append(datum)
                        elif self.var_data_type[index].name == "date":
                            try:
                                date = d_parse.parse(datum)
                            except d_parse.ParserError:
                                date = "N/A"
                            data.append(date)
                        elif self.var_data_type[index].name == "missing":
                            data.append(datum)
                        else:
                            raise ValueError("IMPOSSIBLE! Invalid data type: {}".format(self.var_data_type[index]))
                self.whole_data_list.append(data)

        self.whole_data_array = m.Matrix(0, 0, self.whole_data_list)
        self.data_array = m.Matrix(self.whole_data_array.rows, 0)
        print("Data extracted from file. \nNow processing data...\n")
        for index, var_type in enumerate(self.var_data_type):
            if var_type.name != "missing":
                self.data_array.r_append(
                    m.Matrix(0, 0, list(map(lambda x: [x], self.whole_data_array.get_col(index)))))

        # print("Whole data: \n{}".format(self.whole_data_array))
        # print("Data: \n{}".format(self.data_array))
        self.data = self.data_array.to_numpy()
        self.headers = []
        self.header2col = {}
        self.col2header = {}
        new_index = 0
        for index, header in enumerate(self.whole_headers):
            if self.var_data_type[index].name != "missing":
                # print(self.var_data_type[index])
                # print(header)
                self.headers.append(header)
                self.header2col[header] = new_index
                self.col2header[new_index] = header
                new_index += 1

        print("Data processing complete!\n")

    def __str__(self):
        """toString method

        (For those who don't know, __str__ works like toString in Java...In this case, it's what's called to determine
        what gets shown when a `Data` object is printed.)

        Returns:
        -----------
        str. A nicely formatted string representation of the data in this Data object.
            Only show, at most, the 1st 5 rows of data
            See the test code for an example output.

        NOTE: It is fine to print out int-coded categorical variables (no extra work compared to printing out numeric data).
        Printing out the categorical variables with string levels would be a small extension.
        """
        sizes = []
        data_types = []
        out_list = ["┌"]
        for index, word in enumerate(self.headers):
            temp_size = len(word) + 2
            data_type = self.var_data_type[self.whole_header2col[self.col2header[index]]]
            data_types.append(data_type)
            if data_type.name == "categorical":
                for category in self.cats2levels[word]:
                    new_size = len(category) + 5 + len(str(self.cats2level_dicts[word][category]))
                    if new_size > temp_size:
                        temp_size = new_size
            else:
                for entry in self.data[:, self.header2col[word]]:
                    if len(str(entry)) + 2 > temp_size:
                        temp_size = len(str(entry)) + 2
            sizes.append(temp_size)
            out_list.append(temp_size * "─")
            out_list.append("┬")
        out_list.pop(-1)
        out_list.append("┐\n│")
        for index, word in enumerate(self.headers):
            size = sizes[index]
            sizer = '{:^' + str(size) + '.' + str(size) + '}'
            out_list.append(sizer.format(str(word)) + "│")
        out_list.append("\n├")
        for s in sizes:
            out_list.append(s * "─")
            out_list.append("┼")
        out_list.pop(-1)
        out_list.append("┤\n")
        rows = self.data_array.row_set()
        row_count = len(rows)
        order = math.floor(math.log10(row_count))
        print("Row count: {}".format(row_count))
        print("Order: {}".format(order))
        for ind, row in enumerate(rows):
            if ind % (1000 * (math.pow(10, order - 5))) == 0 and row_count >= 10000:
                ratio = ind / row_count
                print("String output {:.2%}".format(ratio))
            out_list.append("│")
            for index, entry in enumerate(row):
                data_type = data_types[index]
                fill = entry
                size = sizes[index]
                sizer = '{:^' + str(size) + '.' + str(size) + '}'
                if data_type.name == "categorical":
                    fill = self.levels2cats_dicts[self.col2header[index]][int(entry)] + " (" + str(entry) + ")"
                out_list.append(sizer.format(str(fill)) + "│")
            out_list.append("\n")
        out_list.append("└")
        for s in sizes:
            out_list.append(s * "─")
            out_list.append("┴")
        out_list.pop(-1)
        out_list.append("┘\n")
        return "".join(out_list)

    def get_headers(self):
        """Get list of header names (all variables)

        Returns:
        -----------
        Python list of str.
        """
        return self.headers

    def get_mappings(self):
        """Get method for mapping between variable name and column index

        Returns:
        -----------
        Python dictionary. str -> int
        """
        return self.header2col

    def get_cat_level_mappings(self):
        """Get method for mapping between categorical variable names and a list of the respective unique level strings.

        Returns:
        -----------
        Python dictionary. str -> list of str
        """
        return self.cats2levels

    def get_num_dims(self):
        """Get method for number of dimensions in each data sample

        Returns:
        -----------
        int. Number of dimensions in each data sample. Same thing as number of variables.
        """
        return len(self.headers)

    def get_num_samples(self):
        """Get method for number of data points (samples) in the dataset

        Returns:
        -----------
        int. Number of data samples in dataset.
        """
        return self.data_array.rows

    def get_sample(self, rowInd):
        """Gets the data sample at index `rowInd` (the `rowInd`-th sample)

        Returns:
        -----------
        ndarray. shape=(num_vars,) The data sample at index `rowInd`
        """
        return self.data_array.get_row(rowInd)

    def get_header_indices(self, headers):
        """Gets the variable (column) indices of the str variable names in `headers`.

        Parameters:
        -----------
        headers: Python list of str. Header names to take from self.data

        Returns:
        -----------
        Python list of nonnegative ints. shape=len(headers). The indices of the headers in `headers` list.
        """
        output = []
        for header in headers:
            output.append(self.header2col[header])
        return output

    def get_all_data(self):
        """Gets a copy of the entire dataset

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_data_samps, num_vars). A copy of the entire dataset.
            NOTE: This should be a COPY, not the data stored here itself. This can be accomplished with numpy's copy
            function.
        """
        return self.data.copy()

    def head(self):
        """Return the 1st five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). 1st five data samples.
        """
        return self.data[:5, :]

    def tail(self):
        """Return the last five data samples (all variables)

        (Week 2)

        Returns:
        -----------
        ndarray. shape=(5, num_vars). Last five data samples.
        """
        return self.data[-5:, :]

    def limit_samples(self, start_row, end_row):
        """Update the data so that this `Data` object only stores samples in the contiguous range:
            `start_row` (inclusive), end_row (exclusive)
        Samples outside the specified range are no longer stored.

        (Week 2)

        """
        setter = self.data[start_row:end_row, :]
        self.data = setter

    def select_data(self, headers, rows=None):
        """Return data samples corresponding to the variable names in `headers`.
        If `rows` is empty, return all samples, otherwise return samples at the indices specified by the `rows` list.

        (Week 2)

        For example, if self.headers = ['a', 'b', 'c'] and we pass in header = 'b', we return column #2 of self.data.
        If rows is not [] (say =[0, 2, 5]), then we do the same thing, but only return rows 0, 2, and 5 of column #2.

        Parameters:
        -----------
            headers: Python list of str. Header names to take from self.data
            rows: Python list of int. Indices of subset of data samples to select. Empty list [] means take all rows.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, len(headers)) if rows=[]
                 shape=(len(rows), len(headers)) otherwise
            Subset of data from the variables `headers` that have row indices `rows`.

        Hint: For selecting a subset of rows from the data ndarray, check out np.ix_
        """
        if rows is None:
            rows = []
        temp_array = self.data
        if len(rows) != 0:
            temp_array = self.data[[rows], :][0]
        return temp_array[:, self.get_header_indices(headers)]


def data2str(data: np.ndarray, headers: List[str], cats2level_dicts: Dict[str, Dict[str, int]],
             var_data_type: List[dT],
             whole_header2col: Dict[str, int], header2col: Dict[str, int]):
    """toString method

    (For those who don't know, __str__ works like toString in Java...In this case, it's what's called to determine
    what gets shown when a `Data` object is printed.)

    Returns:
    -----------
    str. A nicely formatted string representation of the data in this Data object.
        Only show, at most, the 1st 5 rows of data
        See the test code for an example output.

    NOTE: It is fine to print out int-coded categorical variables (no extra work compared to printing out numeric data).
    Printing out the categorical variables with string levels would be a small extension.
    """
    data_output = data.tolist()

    data_array = m.Matrix(0, 0, data_output)
    new_dict = {}
    for index, word in enumerate(headers):
        new_dict[word] = index
    header2col = new_dict
    new_list = []
    for index, word in enumerate(headers):
        new_list.append(var_data_type[whole_header2col[word]])
    var_data_type = new_list
    col2header = {}
    for index, word in enumerate(headers):
        col2header[index] = word

    flip_map = map(lambda x: [x, dict(zip(cats2level_dicts[str(x)].values(), cats2level_dicts[str(x)].keys()))],
                   list(cats2level_dicts.keys()))

    lc_keys = []
    lc_vals = []

    for entry in flip_map:
        lc_keys.append(entry[0])
        lc_vals.append(entry[1])

    levels2cats_dicts = dict(zip(lc_keys, lc_vals))

    sizes = []
    data_types = []
    out_list = ["┌"]
    for index, word in enumerate(headers):
        temp_size = len(word) + 2
        data_type = var_data_type[index]
        data_types.append(data_type)
        if data_type.name == "categorical":
            for category in cats2level_dicts[word].keys():
                new_size = len(category) + 5 + len(str(cats2level_dicts[word][category]))
                if new_size > temp_size:
                    temp_size = new_size
        else:
            for entry in data[:, header2col[word]]:
                if len(str(entry)) + 2 > temp_size:
                    temp_size = len(str(entry)) + 2
        sizes.append(temp_size)
        out_list.append(temp_size * "─")
        out_list.append("┬")
    out_list.pop(-1)
    out_list.append("┐\n│")
    for index, word in enumerate(headers):
        size = sizes[index]
        sizer = '{:^' + str(size) + '.' + str(size) + '}'
        out_list.append(sizer.format(str(word)) + "│")
    out_list.append("\n├")
    for s in sizes:
        out_list.append(s * "─")
        out_list.append("┼")
    out_list.pop(-1)
    out_list.append("┤\n")
    rows = data_array.row_set()
    row_count = len(rows)
    order = math.floor(math.log10(row_count))
    print("Row count: {}".format(row_count))
    print("Order: {}".format(order))
    for ind, row in enumerate(rows):
        row = list(map(lambda x: row[header2col[x]], headers))
        if ind % (1000 * (math.pow(10, order - 5))) == 0 and row_count >= 10000:
            ratio = ind / row_count
            print("String output {:.2%}".format(ratio))
        out_list.append("│")
        for index, entry in enumerate(row):
            data_type = data_types[index]
            fill = entry
            size = sizes[index]
            sizer = '{:^' + str(size) + '.' + str(size) + '}'
            if data_type.name == "categorical":
                fill = levels2cats_dicts[col2header[index]][int(entry)] + " (" + str(int(entry)) + ")"
            out_list.append(sizer.format(str(fill)) + "│")
        out_list.append("\n")
    out_list.append("└")
    for s in sizes:
        out_list.append(s * "─")
        out_list.append("┴")
    out_list.pop(-1)
    out_list.append("┘\n")
    return "".join(out_list)


def data2str_source(data: np.ndarray, data_source: Data):
    """toString method

    (For those who don't know, __str__ works like toString in Java...In this case, it's what's called to determine
    what gets shown when a `Data` object is printed.)

    Returns:
    -----------
    str. A nicely formatted string representation of the data in this Data object.
        Only show, at most, the 1st 5 rows of data
        See the test code for an example output.

    NOTE: It is fine to print out int-coded categorical variables (no extra work compared to printing out numeric data).
    Printing out the categorical variables with string levels would be a small extension.
    """
    data_output = data.tolist()

    data_array = m.Matrix(0, 0, data_output)
    headers = data_source.headers
    var_data_type = data_source.var_data_type
    header2col = data_source.header2col
    col2header = data_source.col2header
    cats2level_dicts = data_source.cats2level_dicts
    levels2cats_dicts = data_source.levels2cats_dicts

    sizes = []
    data_types = []
    out_list = ["┌"]
    for index, word in enumerate(headers):
        temp_size = len(word) + 2
        data_type = var_data_type[index]
        data_types.append(data_type)
        if data_type.name == "categorical":
            for category in cats2level_dicts[word].keys():
                new_size = len(category) + 5 + len(str(cats2level_dicts[word][category]))
                if new_size > temp_size:
                    temp_size = new_size
        else:
            for entry in data[:, header2col[word]]:
                if len(str(entry)) + 2 > temp_size:
                    temp_size = len(str(entry)) + 2
        sizes.append(temp_size)
        out_list.append(temp_size * "─")
        out_list.append("┬")
    out_list.pop(-1)
    out_list.append("┐\n│")
    for index, word in enumerate(headers):
        size = sizes[index]
        sizer = '{:^' + str(size) + '.' + str(size) + '}'
        out_list.append(sizer.format(str(word)) + "│")
    out_list.append("\n├")
    for s in sizes:
        out_list.append(s * "─")
        out_list.append("┼")
    out_list.pop(-1)
    out_list.append("┤\n")
    rows = data_array.row_set()
    row_count = len(rows)
    order = math.floor(math.log10(row_count))
    print("Row count: {}".format(row_count))
    print("Order: {}".format(order))
    for ind, row in enumerate(rows):
        if ind % (1000 * (math.pow(10, order - 5))) == 0 and row_count >= 10000:
            ratio = ind / row_count
            print("String output {:.2%}".format(ratio))
        out_list.append("│")
        for index, entry in enumerate(row):
            data_type = data_types[index]
            fill = entry
            size = sizes[index]
            sizer = '{:^' + str(size) + '.' + str(size) + '}'
            if data_type.name == "categorical":
                fill = levels2cats_dicts[col2header[index]][int(entry)] + " (" + str(int(entry)) + ")"
            out_list.append(sizer.format(str(fill)) + "│")
        out_list.append("\n")
    out_list.append("└")
    for s in sizes:
        out_list.append(s * "─")
        out_list.append("┴")
    out_list.pop(-1)
    out_list.append("┘\n")
    return "".join(out_list)
