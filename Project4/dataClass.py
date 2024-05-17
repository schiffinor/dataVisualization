"""
dataClass.py
Reads CSV files, stores data, access/filter data by variable name
Added some extra features such as type handling and per type data validation for each type.
Roman Schiffino
CS 251: Data Analysis and Visualization
Spring 2024
"""

import math
from typing import List, Dict

import dateutil.parser as d_parse
import numpy as np
import pandas
import matrix as m
from dataTypes import DataTypes as dT
from dataTypesTrim import DataTypesTrim as dTT


class Data:
    """
    Represents data read in from .csv files
    """

    def __init__(self, filepath: str = None, headers: List[str] = None, data: np.ndarray = None,
                 dFrame: pandas.DataFrame = None, header2col: Dict[str, int] = None, cats2levels: List[str] = None,
                 allDataTypes=False, compatMode=False):
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
        self.data_copy = data.copy() if data is np.ndarray else data
        self.header2col = header2col
        self.whole_header2col = None
        # I love ternary operators if that wasn't immediately obvious. (Java's is better.)
        self.col2header = None if header2col is None else dict(zip(header2col.values(), header2col.keys()))
        self.whole_col2header = None
        # Basically I assume that pythons dict is a hashmap. Hashmap is more efficient than a list.
        self.cats2levels = {} if cats2levels is None else cats2levels
        self.cats2level_dicts = {} if cats2levels is None else None
        self.levels2cats_dicts = {} if cats2levels is None else None
        self.allDataTypes = allDataTypes
        # To meet the requirements of test files this was added. Basically restricts possible data types.
        self.dTRef = dT if allDataTypes else dTT
        # Also to meet the requirements of test files this was added.
        # My base code is quite robust and handles a lot of different stuff by default.
        # This is a feature to limit that handling. Causes error if all data types are missing.
        # Otherwise, will process incorrectly formatted csvs.
        self.compatMode = compatMode

        if filepath is not None:
            self.read(filepath)
        if dFrame is not None:
            self.dTRef = dT
            self.data = dFrame.to_numpy()
            # print(f"Data Types: {self.data.dtype}")
            # print(f"Data Types: {self.data[0].dtype}")
            # print(f"Data Types: {self.data[0][0].dtype}")
            self.headers = list(dFrame.columns)
            dataTypes = list(dFrame.dtypes)
            self.var_data_type = [(self.dTRef.numeric if np.issubdtype(dType, np.inexact) else
                                   (self.dTRef.date if dType == np.datetime64 else
                                    (self.dTRef.categorical if np.issubdtype(dType, np.integer) else
                                     (self.dTRef.string if (dType.type is np.string_ or dType.type is np.str_) else
                                      self.dTRef.missing)))) for dType in dataTypes]
            self.header2col = dict(zip(self.headers, range(len(self.headers))))
            self.col2header = dict(zip(range(len(self.headers)), self.headers))
            self.whole_headers = self.headers
            self.whole_header2col = self.header2col
            self.whole_col2header = self.col2header
            self.data_array = m.Matrix(0, 0, self.data)
            self.whole_data_array = self.data_array
            self.data_copy = self.data.copy()

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
        # Validation for filepath.
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
            # Process to out-load stripped lines to list such that we can close file.
            raw_file_lines = self.file.readlines()
            file_lines = []
            for raw_line in raw_file_lines:
                line = raw_line.strip("\n")
                file_lines.append(line)
            self.file.close()

            # Split at commas.
            raw_headers = file_lines[0].split(',')
            headers = []
            for raw_var_name in raw_headers:
                var_name = raw_var_name.strip()
                headers.append(var_name)
            self.whole_headers = headers
            # Test line.
            # print("Headers: {}".format(self.headers))

            # Assuming that the python dictionary is based on a hashmap,
            # having two reflexive dictionaries is more time efficient than using the __index__ operator on a list.
            # Also, the whole_var constructions are suc that we can store the entirety of the data for general type,
            # then refine based on selected accepted data-types.
            self.whole_header2col = dict(zip(self.whole_headers, range(len(self.whole_headers))))
            self.whole_col2header = dict(zip(range(len(self.whole_headers)), self.whole_headers))
            raw_data_types = file_lines[1].split(',')
            data_types = []
            for index, raw_data_type in enumerate(raw_data_types):
                data_type = raw_data_type.strip()
                # Test line.
                # print("Data type: {}".format(data_type))

                # Verify whether data type is accepted.
                if data_type not in self.dTRef.__members__:
                    print("Invalid data type: {}\nIgnoring Column.\n".format(data_type))
                    data_types.append(self.dTRef["missing"])
                else:
                    data_types.append(self.dTRef[data_type])
            # List of data types corresponding to index of corresponding header in whole_headers2col.
            self.var_data_type = data_types

            # Ensures behavior that test code 2 crashes on invalid data types.
            # Basically, if all data types are missing,
            # then raise an error if and only if var_data_type is not empty and not in compat mode.
            if all(d_type.name == "missing" for d_type in self.var_data_type) and len(
                    self.var_data_type) > 0 and not self.compatMode:
                raise ValueError("All data types invalid.\n"
                                 "Input file likely missing data types.\n"
                                 "Second line must list data types."
                                 "Second line processed: {}".format(file_lines[1]))
            # Test lines.
            # print("Data types: {}".format(self.var_data_type))
            # print("Data types: {}".format(list(map(lambda var: var.name, self.var_data_type))))

            # Here again because hashmaps are better than lists, a bijective mapping is what were implementing here.
            # The four variables below are all dictionaries.
            # The base of all four structures below is a map from the set of headers to some other collection.
            # In self.cats2levels each element of the value set is a list.
            # self.cats2levels maps to a list of the levels.
            # In both self.cats2level_dicts and self.levels2cats_dicts is instead another dictionary.
            # self.cats2level_dicts maps to a map from the set of categories to the set of levels.
            # self.levels2cats_dicts maps to a map from the set of levels to the set of categories.
            # This structure is considerably less space efficient than a list. Yet, it is considerably more time efficient.
            # As hashmap lookup is O(1) and list lookup is O(n).
            for index, datum in enumerate(self.var_data_type):
                if datum.name == "categorical":
                    self.cats2levels[self.whole_headers[index]] = []
                    self.cats2level_dicts[self.whole_headers[index]] = {}
                    self.levels2cats_dicts[self.whole_headers[index]] = {}

            # This behemoth is the data interpretation, validation, and categorization structure.
            # It's large and a bit complex but it handles a lot.
            self.whole_data_list = []
            for row_index, line in enumerate(file_lines[2:]):
                # Split at commas.
                raw_data = line.split(',')
                # Initiate row of data as empty list.
                data = []
                for index, raw_datum in enumerate(raw_data):
                    # Allows for string data types to include leading and trailing whitespace, which is otherwise removed.
                    # This allows for strings to be just sequences of white space which could be handy.
                    datum = raw_datum.strip() if self.var_data_type[index].name != "string" else raw_datum

                    # Handles missing data points.
                    if datum == "":
                        # Empty numeric handling.
                        if self.var_data_type[index].name == "numeric":
                            data.append(np.nan)
                        # Empty categorical handling.
                        elif self.var_data_type[index].name == "categorical":
                            # Category defined as missing.
                            category = "Missing"
                            # Header determination via index of corresponding header in whole_headers2col.
                            temp_header = self.whole_col2header[index]
                            # Corresponding self.cats2levels and self.cats2level_dicts maps.
                            temp_list = self.cats2levels[temp_header]
                            temp_dict = self.cats2level_dicts[temp_header]
                            # Handling for when "Missing" is not in the category list.
                            if category not in temp_list:
                                # Add missing to category list.
                                temp_list.append(category)
                                # Calculate index of most recently added entry.
                                index = len(temp_list) - 1
                                # Update category dictionary based on new entry.
                                self.cats2level_dicts[temp_header][temp_list[index]] = index
                                self.levels2cats_dicts[temp_header][index] = category
                                # Updates temp dict and self.cats2level.
                                # (I'm not sure if python stores a copy or a memory reference when defining var by another var.)
                                temp_dict = self.cats2level_dicts[temp_header]
                                self.cats2levels[temp_header] = temp_list
                            # Adds datum to data row list.
                            data.append(temp_dict[category])
                        # Empty string handling.
                        elif self.var_data_type[index].name == "string":
                            data.append(datum)
                        # Empty date handling.
                        elif self.var_data_type[index].name == "date":
                            data.append("N/A")
                        # Empty missing handling.
                        elif self.var_data_type[index].name == "missing":
                            data.append(datum)
                        # Default handling.
                        else:
                            raise ValueError("IMPOSSIBLE! Invalid data type: {}".format(self.var_data_type[index]))

                    # Standard data handling.
                    else:
                        # Numeric handling.
                        if self.var_data_type[index].name == "numeric":
                            # Try and cast to float.
                            try:
                                number = float(datum)
                            # Catch overflow errors.
                            except OverflowError:
                                number = float("inf")
                                print("Overflow Error: {} is too large.".format(datum))
                            # Catch value errors.
                            except ValueError:
                                number = np.nan
                                print("Value Error: {} is not a number.".format(datum))
                            data.append(number)
                        # Categorical handling.
                        elif self.var_data_type[index].name == "categorical":
                            # Header determination via index of corresponding header in whole_headers2col.
                            temp_header = self.whole_col2header[index]
                            # Corresponding self.cats2levels and self.cats2level_dicts maps.
                            temp_list = self.cats2levels[temp_header]
                            temp_dict = self.cats2level_dicts[temp_header]
                            # Handling for when datum category is not in the category list.
                            if datum not in temp_list:
                                # Add missing to category list.
                                temp_list.append(datum)
                                # Calculate index of most recently added entry.
                                index = len(temp_list) - 1
                                # Update category dictionary based on new entry.
                                self.cats2level_dicts[temp_header][temp_list[index]] = index
                                self.levels2cats_dicts[temp_header][index] = datum
                                # Updates temp dict and self.cats2level.
                                # (I'm not sure if python stores a copy or a memory reference when defining var by another var.)
                                temp_dict = self.cats2level_dicts[temp_header]
                                self.cats2levels[temp_header] = temp_list
                            # Adds datum to data row list.
                            data.append(temp_dict[datum])
                        # String type handling.
                        elif self.var_data_type[index].name == "string":
                            data.append(datum)
                        # Date type handling.
                        elif self.var_data_type[index].name == "date":
                            # Try and cast to datetime.
                            try:
                                date = d_parse.parse(datum)
                            # Catch non-date errors.
                            except d_parse.ParserError:
                                date = "N/A"
                            data.append(date)
                        # Missing type handling.
                        elif self.var_data_type[index].name == "missing":
                            data.append(datum)
                        # Default handling.
                        else:
                            raise ValueError("IMPOSSIBLE! Invalid data type: {}".format(self.var_data_type[index]))
                # Append row of data to whole data list.
                self.whole_data_list.append(data)

        # Create matrix from the whole data list. Matrix is built on an implementation I wrote a couple of years back.
        # Originally was coded such that matrix operations treated matrices as immutable.
        # However, I did a complete conversion of all the methods within the matrix class that I used in this to make them run faster.
        # This coupled with some other performance improvements resulted in a 138.25 times speedup for print_austin function.
        # Regardless this structure parses the whole_data_array into a matrix consisting of only non-missing data types.
        # That is unless the data is in compat mode.
        self.whole_data_array = m.Matrix(0, 0, self.whole_data_list)
        # Create empty matrix for data.
        self.data_array = m.Matrix(self.whole_data_array.rows, 0)
        print("Data extracted from file. \nNow processing data...\n")
        if self.compatMode:
            self.data_array = self.whole_data_array
        else:
            for index, var_type in enumerate(self.var_data_type):
                # Append non-missing data to data array.
                if var_type.name != "missing":
                    # Pretty simple mapping expression with lambda that composes a column vector via a map of the xth index of each column.
                    self.data_array.r_append(
                        m.Matrix(0, 0, list(map(lambda x: [x], self.whole_data_array.get_col(index)))))

        # Test code.
        # print("Whole data: \n{}".format(self.whole_data_array))
        # print("Data: \n{}".format(self.data_array))

        # Handy function I added for ease in my matrix class.
        # Completely unnecessary as it makes my module less portable, but whatever, convenience.
        self.data = self.data_array.to_numpy()
        self.data_copy = self.data.copy()
        # Defines all our referencable values in terms of indexes of the new matrix.
        self.headers = []
        self.header2col = {}
        self.col2header = {}
        # Base index that will be incremented for each header with a non-missing datatype.
        new_index = 0
        for index, header in enumerate(self.whole_headers):
            if self.var_data_type[index].name != "missing":
                # Test code.
                # print(self.var_data_type[index])
                # print(header)
                # Appends header to headers list.
                self.headers.append(header)
                # Creates self.header2col and self.col2header dictionaries.
                # Once again, reflected dictionaries for speed.
                self.header2col[header] = new_index
                self.col2header[new_index] = header
                # Increments new_index by 1 for each header
                new_index += 1
        # Little disclaimer to the user.
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

        My Notes:
        OK, so I kind of went over the top with this one. This may have been a little more trouble than it was worth.
        However, it was pretty cool.

        What we do here is generate a very ornately formatted string representation of the data table.
        We have use UTF-8 encoding here in order to make use of the table construction characters.

        Regardless, this was previously a major lag source. The print_austin function was previously quite a bit slower.
        From my experience in Java, and the test code I had baked into this code,
        I had some idea that the lag was likely a result of string concatenation. Thus, I did some research.
        Apparently, string in python are immutable, thus when we concatenate a string using the + operator (__add__()),
        the time complexity is O(n^2).
        As such I checked a comparison of various concatenation methods and came to the result that,
        while the overhead on the str.join() method was greater than the other methods,
        resulting in slower performance for smaller sets of strings,
        for sets of strings with cardinality greater than 10,000 or so,
        the increased speed of the join method resulted in drastically more rapid processing.
        Thus, I rewrote this method to use the str.join() method.
        Every instance of out_string += "~" was replaced with an instance of out_list.append("~").
        Then instead of returning out_string we return "".join(out_list).
        The performance was drastically faster. In fact, it was about 46.87 times faster.
        """
        row_stop = 5
        # Initialize list storing maximum of set of length of string representation for all entries in each column.
        sizes = []
        # Initialize list storing data type for each column.
        data_types = []
        # Initialize list storing all strings to be concatenated.
        out_list = ["┌"]

        # Determine maximum of length of string representations for all entries in each column.
        # This is the most important step in all the formatting as the spacing and all the boundary alignments depend on this.
        for index, word in enumerate(self.headers):
            # By default, temp_size is the length of the header plus two.
            temp_size = len(word) + 2
            # Determine data type for each header by mapping reference index to header,
            # then mapping that header to the actual column in the whole_data_array,
            # then retrieving data type corresponding to that column index.
            data_type = self.var_data_type[self.whole_header2col[self.col2header[index]]]
            # Append to local datatype reference list.
            data_types.append(data_type)
            # If data type is categorical, we will be representing it in the following format:
            # category.name (category.level)
            # Thus the length of this string representation is the length of the category name, plus 5,
            # 3 spaces and 2 parentheses, plus the length of the levels string representation.
            if data_type.name == "categorical":
                for category in self.cats2levels[word]:
                    new_size = len(category) + 5 + len(str(self.cats2level_dicts[word][category]))
                    # If the length of this string representation is greater than the current temp size we update the temp size.
                    if new_size > temp_size:
                        temp_size = new_size
            # For all other data types, the length of this string representation is just that plus two (for spaces).
            # As such we update the temp size if and only if len(str(entry)) + 2  is greater than the current temp size.
            else:
                for entry in self.data[:, self.header2col[word]]:
                    if len(str(entry)) + 2 > temp_size:
                        temp_size = len(str(entry)) + 2
            # Append temp_size (maximum length of string representation) to local list.
            sizes.append(temp_size)
            # Border framing.
            out_list.append(temp_size * "─")
            out_list.append("┬")
        # Remove excess "┬".
        out_list.pop(-1)
        out_list.append("┐\n│")

        # Here we create our string representation of the data in the data table.
        # This loop handles header row of the data table and manages spacing.
        for index, word in enumerate(self.headers):
            # Get size.
            size = sizes[index]
            # Create a truncation and alignment formatting string.
            # These are interpreted as strings, so they must be composed first.
            # This particular string instructs the formatter to make sure the string is center aligned,
            # and exactly size characters long.
            sizer = '{:^' + str(size) + '.' + str(size) + '}'
            out_list.append(sizer.format(str(word)) + "│")
        out_list.append("\n├")
        # Separatory boundary between the variable names and the data table.
        for s in sizes:
            out_list.append(s * "─")
            out_list.append("┼")
        # Remove excess "┼".
        out_list.pop(-1)
        out_list.append("┤\n")
        rows = self.data.tolist()
        row_count = len(rows)
        # Calculates order of the row count for the dataset, ie floor of the base 10 log of the row count.
        order = math.floor(math.log10(row_count))
        # Test code, but useful information, so left in.
        print("Row count: {}".format(row_count))
        print("Order: {}".format(order))
        # This loop handles each row of the data table and manages spacing.
        for ind, row in enumerate(rows):
            # Progress notifications, helpful for the user when printing out massive datasets to make sure program isn't frozen.
            # Notifications only appear if row count is greater than 10,000.
            # Notification rate is determined by the order of the row count.
            # Larger sets will have notifications spaced at larger gaps in row count.
            if ind % (1000 * (math.pow(10, order - 5))) == 0 and row_count >= 10000:
                ratio = ind / row_count
                # Progress reported as a percentage with 2 decimal places.
                print("String output {:.2%}".format(ratio))
            out_list.append("│")
            # This loop handles each entry of the row and manages string representation of the entry.
            for index, entry in enumerate(row):
                # Determine data type for each entry using local reference list.
                data_type = data_types[index]
                fill = entry
                # Get size.
                size = sizes[index]
                # Create a truncation and alignment formatting string.
                sizer = '{:^' + str(size) + '.' + str(size) + '}'
                # String representation of categorical data.
                if data_type.name == "categorical":
                    # category.name (category.level)
                    fill = self.levels2cats_dicts[self.col2header[index]][int(entry)] + " (" + str(entry) + ")"
                if ind == row_stop and row_stop != 0:
                    fill = "..."
                out_list.append(sizer.format(str(fill)) + "│")
            out_list.append("\n")
            if ind == row_stop and row_stop != 0:
                out_list.append("│")
                ind_d = row_count - 1
                row_d = rows[ind_d]
                for index, entry in enumerate(row_d):
                    # Determine data type for each entry using local reference list.
                    data_type = data_types[index]
                    fill = entry
                    # Get size.
                    size = sizes[index]
                    # Create a truncation and alignment formatting string.
                    sizer = '{:^' + str(size) + '.' + str(size) + '}'
                    # String representation of categorical data.
                    if data_type.name == "categorical":
                        # category.name (category.level)
                        fill = self.levels2cats_dicts[self.col2header[index]][int(entry)] + " (" + str(entry) + ")"
                    out_list.append(sizer.format(str(fill)) + "│")
                out_list.append("\n")
                break
        # Create lower border for table.
        out_list.append("└")
        for s in sizes:
            out_list.append(s * "─")
            out_list.append("┴")
        # Remove excess "┴".
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
        return [self.header2col[header] for header in headers]

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
        output = self.get_all_data()
        heads = self.get_header_indices(headers)

        if rows is not None and len(rows) != 0:
            output = output[list(rows)]
        output = output[:, heads]
        return output


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


        My Notes:
        OK, again, I kind of went over the top with this one.
        This is a generalization of out __str__() method in our data structure to arbitrary ndarray given some indexing maps.

        What we do here is generate a very ornately formatted string representation of the data table.
        We have use UTF-8 encoding here in order to make use of the table construction characters.
    """
    # Turns a np.ndarray object into a list.
    data_output = data.tolist()

    # Creates a matrix object from the data_output list.

    # This next section allows us to convert some of the passed in variables into variables that we need but were not provided.
    # This allows us to require less parameter inputs, allowing for my and the users' sanity to be maintained.
    # Create a new temp dict which will be used to store the mapping between headers and columns.
    new_dict = {}
    for index, word in enumerate(headers):
        new_dict[word] = index
    # Assign the new_dict to header2col.
    header2col = new_dict
    # Create a new temp list which will store variable data types corresponding to each header.
    new_list = []
    for index, word in enumerate(headers):
        new_list.append(var_data_type[whole_header2col[word]])
    # Assign the new_list to var_data_type.
    var_data_type = new_list
    # Create a new temp dict which will store the mapping between columns and headers.
    col2header = {}
    for index, word in enumerate(headers):
        col2header[index] = word

    levels2cats_dicts = {k: {v: k for k, v in cats2level_dicts[str(k)].items()} for k in cats2level_dicts.keys()}

    # Initialize list storing maximum of set of length of string representation for all entries in each column.
    sizes = []
    # Initialize list storing data type for each column.
    data_types = []
    # Initialize list storing all strings to be concatenated.
    out_list = ["┌"]
    # Determine maximum of length of string representations for all entries in each column.
    # This is the most important step in all the formatting as the spacing and all the boundary alignments depend on this.
    for index, word in enumerate(headers):
        # By default, temp_size is the length of the header plus two.
        temp_size = len(word) + 2
        # Determine data type for each header by mapping reference index to header,
        # then mapping that header to the actual column in the whole_data_array,
        # then retrieving data type corresponding to that column index.
        data_type = var_data_type[index]
        # Append to local datatype reference list.
        data_types.append(data_type)
        # If data type is categorical, we will be representing it in the following format:
        # category.name (category.level)
        # Thus the length of this string representation is the length of the category name, plus 5,
        # 3 spaces and 2 parentheses, plus the length of the levels string representation.
        if data_type.name == "categorical":
            for category in cats2level_dicts[word].keys():
                new_size = len(category) + 5 + len(str(cats2level_dicts[word][category]))
                # If the length of this string representation is greater than the current temp size we update the temp size.
                if new_size > temp_size:
                    temp_size = new_size
        # For all other data types, the length of this string representation is just that plus two (for spaces).
        # As such we update the temp size if and only if len(str(entry)) + 2  is greater than the current temp size.
        else:
            for entry in data[:, header2col[word]]:
                if len(str(entry)) + 2 > temp_size:
                    temp_size = len(str(entry)) + 2
        # Append temp_size (maximum length of string representation) to local list.
        sizes.append(temp_size)
        # Border framing.
        out_list.append(temp_size * "─")
        out_list.append("┬")
    # Remove excess "┬".
    out_list.pop(-1)
    out_list.append("┐\n│")
    # Here we create our string representation of the data in the data table.
    # This loop handles header row of the data table and manages spacing.
    for index, word in enumerate(headers):
        # Get size.
        size = sizes[index]
        # Create a truncation and alignment formatting string.
        # These are interpreted as strings, so they must be composed first.
        # This particular string instructs the formatter to make sure the string is center aligned,
        # and exactly size characters long.
        sizer = '{:^' + str(size) + '.' + str(size) + '}'
        out_list.append(sizer.format(str(word)) + "│")
    out_list.append("\n├")
    # Separatory boundary between the variable names and the data table.
    for s in sizes:
        out_list.append(s * "─")
        out_list.append("┼")
    # Remove excess "┼".
    out_list.pop(-1)
    out_list.append("┤\n")
    rows = data_output
    row_count = len(rows)
    # Calculates order of the row count for the dataset, ie floor of the base 10 log of the row count.
    order = math.floor(math.log10(row_count))
    # Test code, but useful information, so left in.
    print("Row count: {}".format(row_count))
    print("Order: {}".format(order))
    # This loop handles each row of the data table and manages spacing.
    for ind, row in enumerate(rows):
        # Progress notifications, helpful for the user when printing out massive datasets to make sure program isn't frozen.
        # Notifications only appear if row count is greater than 10,000.
        # Notification rate is determined by the order of the row count.
        # Larger sets will have notifications spaced at larger gaps in row count.
        row = list(map(lambda x: row[header2col[x]], headers))
        if ind % (1000 * (math.pow(10, order - 5))) == 0 and row_count >= 10000:
            ratio = ind / row_count
            # Progress reported as a percentage with 2 decimal places.
            print("String output {:.2%}".format(ratio))
        out_list.append("│")
        # This loop handles each entry of the row and manages string representation of the entry.
        for index, entry in enumerate(row):
            # Determine data type for each entry using local reference list.
            data_type = data_types[index]
            fill = entry
            # Get size.
            size = sizes[index]
            # Create a truncation and alignment formatting string.
            sizer = '{:^' + str(size) + '.' + str(size) + '}'
            # String representation of categorical data.
            if data_type.name == "categorical":
                # category.name (category.level)
                fill = levels2cats_dicts[col2header[index]][int(entry)] + " (" + str(int(entry)) + ")"
            out_list.append(sizer.format(str(fill)) + "│")
        out_list.append("\n")
    # Create lower border for table.
    out_list.append("└")
    for s in sizes:
        out_list.append(s * "─")
        out_list.append("┴")
    # Remove excess "┴".
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

    Same as above, just self extracting data from an accompanying Data object.
    """
    # Extract data from Data object.
    headers = data_source.headers
    var_data_type = data_source.var_data_type
    whole_header2col = data_source.whole_header2col
    header2col = data_source.header2col
    cats2level_dicts = data_source.cats2level_dicts
    return data2str(data, headers, cats2level_dicts, var_data_type, whole_header2col, header2col)
