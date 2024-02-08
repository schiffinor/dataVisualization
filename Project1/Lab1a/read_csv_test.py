"""read_csv_test.py
Test code for read_csv function in csv_reader.py
Oliver W. Layton
CS 251: Data Analysis and Visualization
Lab 1a
Fall 2023
"""
import csv_reader


def test_read_csv(filepath):
    data = csv_reader.read_csv(filepath=filepath)

    if data is None:
        print('Noticed your data list of lists variable is not defined. Did you forget to return it?')
        exit()

    print('Your list of lists (data) based on the CSV file is:')
    print(data)
    print('and it should look like:')
    print("[['hi', 'monkey', 'eek eek'], ['there', 'penguin', 'waddle waddle'], ['cs251', 'kangaroo', 'boing boing'], ['!', 'pig', 'oink oink']]")


if __name__ == '__main__':
    test_read_csv(filepath='data/lab_test.csv')
