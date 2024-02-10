"""

"""

from data import Data


def main():
    iris_data = Data('data/iris_no_species.csv')
    print(iris_data)
    print(iris_data.head())
    print(iris_data.tail())


if __name__ == '__main__':
    main()
