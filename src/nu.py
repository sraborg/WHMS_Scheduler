from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
from typing import List, Tuple


class AbstractNu(ABC):

    def __init__(self):
        """

        """
        self._x: Tuple[datetime] = None              # Timestamps
        self._y: Tuple[float] = None              # Values
        self._model = None
        self._f = None
        self._min_regression_value = 0
        self._invalid_time_value = -0.1

    @abstractmethod
    def fit_model(self, values: List[Tuple[datetime, int]]):
        """

        :param values:
        :return:
        """
        self._x, self._y = zip(*values)              # Time, Values

    def eval(self, x):
        """

        :param x:
        :return:
        """
        if x < min(self._x) or x > max(self._x):    # Out of scheduling window (too early/late)
            return self._invalid_time_value
        elif self._f(x) < 0:                        # Fixing negative values returned by regression
            return self._min_regression_value
        else:
            return self._f(x)

    @abstractmethod
    def name(self) -> str:
        pass

    def max_value(self):
        return max(self._y)

class NuRegression(AbstractNu):

    def __init__(self):
        """

        """
        super().__init__()

    def fit_model(self, values: List[Tuple[datetime, int]]):
        """

        :param values:
        :return:
        """
        super().fit_model(values)
        rank = len(self._y) - 2
        self._model = np.polyfit(list(self._x), list(self._y), rank)
        self._f = np.poly1d(self._model)

    def name(self) -> str:
        return "REGRESSION"


class NuConstant(AbstractNu):

    def __init__(self, **kwargs):
        super().__init__()
        self._value = kwargs.get("constant", 1)

    def fit_model(self, values: List[Tuple[datetime, int]]):
        """

        :param values:
        :return:
        """
        pass

    def eval(self, x):
        """

        :param x:
        :return:
        """
        return self._value

    def name(self) -> str:
        return "CONSTANT"


class NuFactory:

    @staticmethod
    def get_nu(nu_type: str, **kwargs):
        nu = None

        if nu_type.upper() == "REGRESSION":
            nu = NuRegression()
        elif nu_type.upper() == "CONSTANT":
            nu = NuConstant(**kwargs)
        else:
            raise Exception("Invalid Analysis Type")

        return nu

    @classmethod
    def regression(cls):
        return NuRegression()

    @classmethod
    def constant(cls, **kwargs):
        return NuConstant(**kwargs)