"""csv_reader.py
Reads in data from .csv files
YOUR NAME HERE
CS 251: Data Analysis and Visualization
Lab 1a
"""


def read_csv(filepath):
    """Reads and returns the data from a CSV file located at `filepath`.

    Parameters:
    -----------
    filepath: str. Path to the .csv file to be read in.

    Returns:
    -----------
    List of lists. The data loaded from the .csv file.

    ----------------------------------------------------------------------------
    Example:
    For a .csv file that looks like:

    a,b,c
    1,2,3

    The corresponding list of lists that this function should return looks like:
    [[a, b, c], [1, 2, 3]]
    ----------------------------------------------------------------------------

    TODO:
    Write code below that does what the docstring above states (i.e. read in the .csv file, organize the data as a
    list of lists, return the list of lists).

    NOTE:
    - You should only use standard Python to implement this method. Do not import other modules.
    - Remember that Python has a helpful `split` string method that splits up a string into a list based on a delimiter
    of your choice.
    - There is a helpful string method to remove new line characters.
    - If you are not using a `with` block, don't forget to close the file handle!
    """
    # YOUR CODE HERE (you can delete the pass statement below)
    pass


def read_cat_csv(filepath):
    """Reads in a CSV file containing categorical data located at `filepath`. Codes the imported categorical data using
    ints (0, 1, ...).

    Parameters:
    -----------
    filepath: str. Path to the .csv file to be read in.

    Returns:
    -----------
    List of lists. The data loaded from the .csv file. ONLY contains ints. The ints represent each variables categorical
        levels coded as ints rather than strings.
    Dictionary. The dictionary that contains the mappings between categorical variable names (keys) and the corresponding
        list of unique levels (represented as STRINGS) of each categorical variable (values).

    ----------------------------------------------------------------------------
    Example:
    For a .csv file that looks like:

    a,1,hi
    b,2,hi
    c,2,hi

    The corresponding list of lists that this function should return looks like:
    [[0, 1, 2], [0, 1, 1], [0, 0, 0]]
    and the dictionary should look like (key -> value)
    'var1' -> ['a', 'b', 'c']
    'var2' -> ['1', '2']
    'var3' -> ['hi']
    ----------------------------------------------------------------------------

    TODO:
    Write code below that achieves what the docstring above states.

    NOTE:
    - Assume that the 3 categorical variables in categorical.csv are called and hard-coded as 'name', 'year', 'hobby'.
    We are doing this because the CSV files in today's lab do not have header or types rows. Use these keys in your
    dictionary.
    - You should only use standard Python to implement this method. Do not import other modules.
    - Your code from `read_csv` above should be a helpful starting point.
    - Reviewing your code in dictionary_practice.py should also be helpful.
    """
    # KEEP ME
    # Names of the variables in categorical.csv in the correct order
    var_names = ['name', 'year', 'hobby']

    # YOUR CODE HERE
    # Initialize level dictionary to empty lists...
    
