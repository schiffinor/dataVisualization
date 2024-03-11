from enum import StrEnum


class RegressTypes(StrEnum):
    """
    Represents the data types that can be stored in a `Data` object
    """
    linear, polynomial, exponential, sinusoidal, mixed = ["linear", "polynomial", "exponential", "sinusoidal", "mixed"]

    def __getitem__(self, item):
        type_match = {
            "linear": self.linear,
            "polynomial": self.polynomial,
            "exponential": self.exponential,
            "sinusoidal": self.sinusoidal,
            "mixed": self.mixed
        }
        return type_match[item]

    def __contains__(self, item):
        type_match = {
            "linear": self.linear,
            "polynomial": self.polynomial,
            "exponential": self.exponential,
            "sinusoidal": self.sinusoidal,
            "mixed": self.mixed
        }
        if item in type_match.keys():
            return True
        if item in type_match.values():
            return True
        else:
            return False

