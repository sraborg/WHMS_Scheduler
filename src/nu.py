from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
from typing import List, Tuple


class AbstractNu(ABC):

    def __init__(self):
        self._x = None              # DateTime
        self._y = None              # Values
        self._model = None
        self._f = None

    @abstractmethod
    def fit_model(self, values: List[Tuple[datetime, int]]):
        self._x, self._y = zip(*values)              # Time, Values

    def eval(self, x):
        return self._f(x)


class NuRegression(AbstractNu):

    def __init__(self):
        super().__init__()

    def fit_model(self, values: List[Tuple[datetime, int]]):
        super().fit_model(values)
        self._model = np.polyfit(list(self._x), list(self._y), len(self._y) - 1)
        self._f = np.poly1d(self._model)


class NuFactory:

    @staticmethod
    def get_nu(nu_type: str):
        nu = None

        if nu_type.upper() == "REGRESSION":
            nu = NuRegression()
        else:
            raise Exception("Invalid Analysis Type")

        return nu