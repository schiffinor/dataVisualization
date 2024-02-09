from enum import Enum


class DataTypes(Enum):
    """
    Represents the data types that can be stored in a `Data` object
    """
    numeric, categorical, missing = range(3)
    Enum.member_names_ = ["numeric", "categorical", "missing"]

    def __getitem__(self, item):
        type_match = {
            "numeric": self.numeric,
            "categorical": self.categorical,
            "missing": self.missing
        }
        return type_match[item]

