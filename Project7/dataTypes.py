from enum import StrEnum


class DataTypes(StrEnum):
    """
    Represents the data types that can be stored in a `Data` object
    """
    numeric, categorical, string, date, missing = ["numeric", "categorical", "string", "date", "missing"]

    def __getitem__(self, item):
        type_match = {
            "numeric": self.numeric,
            "categorical": self.categorical,
            "string": self.string,
            "date": self.date,
            "missing": self.missing
        }
        return type_match[item]

