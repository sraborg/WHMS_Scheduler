from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
from typing import List, Tuple


class AbstractNu(ABC):

    def __init__(self):
        self._x = None              # Timestamps
        self._y = None              # Values
        self._model = None
        self._f = None
        self._min_value = -0.001

    @abstractmethod
    def fit_model(self, values: List[Tuple[datetime, int]]):
        self._x, self._y = zip(*values)              # Time, Values

    def eval(self, x):
        if x < min(self._x) or x > max(self._x):
            return self._min_value
        elif self._f(x) < 0:
            return 0
        else:
            return self._f(x)


class NuRegression(AbstractNu):

    def __init__(self):
        super().__init__()

    def fit_model(self, values: List[Tuple[datetime, int]]):
        super().fit_model(values)
        rank = len(self._y) - 2
        self._model = np.polyfit(list(self._x), list(self._y), rank)
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