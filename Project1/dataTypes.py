from enum import Enum


class DataTypes(Enum):
    """
    Represents the data types that can be stored in a `Data` object
    """
    Enum = Enum
    numeric, categorical, string, date, missing = range(5)
    member_names_ = ["numeric", "categorical", "string", "date", "missing"]

    def __getitem__(self, item):
        type_match = {
            "numeric": self.numeric,
            "categorical": self.categorical,
            "string": self.string,
            "date": self.date,
            "missing": self.missing
        }
        return type_match[item]

