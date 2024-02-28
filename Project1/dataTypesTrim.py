from enum import StrEnum


class DataTypesTrim(StrEnum):
    """
    Represents the data types that can be stored in a `Data` object
    """
    numeric, categorical, missing = ["numeric", "categorical", "missing"]

    def __getitem__(self, item):
        type_match = {
            "numeric": self.numeric,
            "categorical": self.categorical,
            "missing": self.missing
        }
        return type_match[item]

