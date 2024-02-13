"""
read_cat_csv_test.py
Test code for read_cat_csv function in csv_reader.py
Oliver W. Layton
CS 251: Data Analysis and Visualization
Lab 1a
Fall 2023
"""
import csv_reader


def test_read_cat_csv(filepath):
    results = csv_reader.read_cat_csv(filepath=filepath)

    if results is None:
        print('Noticed your read_cat_csv function didnt return anything. Did you forget to add a return?')
        exit()

    if len(results) != 2:
        print('Noticed your read_cat_csv function didnt return two things. Did you forget to add a return?')
        exit()

    data, cats2levels = results

    if data is None:
        print('Noticed your data list of lists variable is not defined. Did you forget to return it?')
        exit()

    if cats2levels is None:
        print('Noticed your dictionary is not defined. Did you forget to return it?')
        exit()

    print('Your list of lists (data) based on the CSV file is:')
    print(data)
    print('and it should look like:')
    print("[[0, 0, 0], [1, 1, 1], [2, 1, 0], [1, 0, 2], [0, 2, 3]]")

    print()

    print('Your categorical variable dictionary based on the CSV file is:')
    print(cats2levels)
    print('and it should look like:')
    print("{'name': ['Susie', 'Tom', 'Daisy'], 'year': ['2023', '2022', '2021'], 'hobby': ['Eating ice cream', 'Collecting rubber ducks', 'Laughing uncontrollably', 'Coding upside down']}")


if __name__ == '__main__':
    test_read_cat_csv(filepath='data/categorical.csv')
