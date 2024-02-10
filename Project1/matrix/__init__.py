"""
__init__.py
Roman Schiffino 151B Fall Semester

This is a pretty basic class I created. It basically just lets me store
data in the form of a 2d array or in other words a matrix. Basically wrote
this to make my life a little easier.
"""
import numpy as np


class Matrix:
    """
    Matrix data-type class. Creates a custom data type with a couple custom
    functions. These allow for some utility that really helps with the data.
    """

    def __init__(self, rowCount, columnCount, data=None):
        """
        Initiates the data set. If no data is provided an empty matrix will be created.
        """
        row_list = []
        # Define .rows and .columns attributes.
        self.rows = rowCount
        self.columns = columnCount
        if data is None:
            for curRow in range(self.rows):
                # Creates list which contains the current row.
                col_list = []
                for curCol in range(self.columns):
                    # Populates list.
                    col_list.append(data)
                # Appends row to list of rows.
                row_list.append(col_list)
            # Creates matrix from list.
            self.load = row_list
        if data is not None:
            # Passes data to make matrix.
            self.load = data
            self.rows = len(data)
            self.columns = len(data[0])

    def __str__(self):
        """
        When a function calls for a string representation of the matrix this function provides a string representation of the matrix.
        """
        rows = self.rows
        cols = self.columns
        output = ""
        for x in range(rows):
            output += "["
            for y in range(cols):
                output += str(self.get(x, y))
                output += ", " if y != cols - 1 else ""
            output += "]\n"
        return output

    def get(self, x, y):
        """
        Returns value of matrix at row x, column y.
        """
        return self.load[x][y]

    def get_row(self, x):
        """
        Returns value of matrix at row x, column y.
        """
        return self.load[x]

    def get_col(self, y):
        """
        Returns value of matrix at row x, column y.
        """
        return self.column_set()[y]

    def set(self, x, y, val):
        """
        Sets value of matrix at row x, column y to value.
        """
        self.load[x][y] = val


    def set_row(self, x, val_list):
        """
        Sets value of matrix at row x to value list.
        """
        if x >= self.rows:
            raise ValueError("row index out of range")
        if len(val_list) != self.columns:
            raise ValueError("length of value list does not match number of columns")
        self.load[x] = val_list

    def set_col(self, y, val_list):
        """
        Sets value of matrix at column y to value list.
        """
        if y >= self.columns:
            raise ValueError("column index out of range")
        if len(val_list) != self.rows:
            raise ValueError("length of value list does not match number of rows")
        for colIndex, row in enumerate(self.load):
            row[y] = val_list[colIndex]


    def set_all(self, val):
        """

        """
        data = [[int(val)] * self.columns] * self.rows
        self.load = data

    def r_append(self, other):
        """
        Constructs a matrix where every unit of matrix other is appended to the right of the matrix self.
        """
        self.columns += other.columns
        for index, row in enumerate(self.load):
            self.load[index] = row + other.load[index]

    def l_append(self, other):
        """
        Constructs a matrix where every unit of matrix other is appended to the left of the matrix self.
        """
        self.columns += other.columns
        for index, row in enumerate(self.load):
            self.load[index] = other.load[index] + row

    def u_append(self, other):
        """
        Constructs a matrix where every unit of matrix other is appended above the matrix self.
        """
        self.rows += other.rows
        self.load = other.load + self.load

    def d_append(self, other):
        """
        Constructs a matrix where every unit of matrix other is appended below the matrix self.
        """
        self.rows += other.rows
        self.load = self.load + other.load


    def __mul__(self, other):
        new_matrix = Matrix(int(self.rows), int(other.columns))
        for x in range(new_matrix.rows):
            for y in range(new_matrix.columns):
                list_row = self.load[x]
                list_column = [other.get(i, y) for i in range(other.rows)]
                list_mult = [list_row[i] * list_column[i] for i in range(len(list_column))]
                value = sum(list_mult)
                new_matrix.set(x, y, value)
        return new_matrix

    def __add__(self, other):
        new_matrix = Matrix(int(self.rows), int(self.columns))
        for x in range(new_matrix.rows):
            for y in range(new_matrix.columns):
                value = self.get(x, y) + other.get(x, y)
                new_matrix.set(x, y, value)
        return new_matrix

    def __len__(self):
        return self.rows * self.columns

    def __contains__(self, item):
        for row in self.load:
            for col in row:
                if col == item:
                    return True
        return False


    def row_set(self):
        return self.load



    def column_set(self):
        output = []
        for i in range(self.columns):
            output.append([])
            for j in range(self.rows):
                output[i].append(self.get(j, i))
        return output


    def to_numpy(self):
        output = np.array(self.load)
        return output
