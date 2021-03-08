from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from operator import itemgetter
import numpy as np
from typing import List, Tuple


class AbstractNu(ABC):

    def __init__(self):
        """

        """
        self._values: List[Tuple[float, float]] = None
        self._x: Tuple[datetime] = None              # Timestamps
        self._y: Tuple[float] = None              # Values
        self._model = None
        self._f = None
        self._min_regression_value = 0
        self._invalid_time_value = 0
        self._earliest_start: datetime = None
        self._soft_deadline: datetime = None
        self._hard_deadline: datetime = None

    @property
    def earliest_start(self):
        return self._earliest_start

    @property
    def soft_deadline(self):
        return self._soft_deadline

    @property
    def hard_deadline(self):
        return self._hard_deadline

    @abstractmethod
    def fit_model(self, values: List[Tuple[datetime, float]]):
        """ Takes a list of datetimes/value tuples and determines the following
        (1) earliest_start
        (2) soft_deadline
        (3) heard_deadline

        Also splits the tuples in to corresponding lists: x = datetimes & y = values

        :param values: a list of timestamp/value tuples
        """
        self._values = values
        self._x, self._y = zip(*values)              # Time, Values
        self._earliest_start = datetime.fromtimestamp(min(self._x))
        self._soft_deadline = datetime.fromtimestamp(max(values, key=itemgetter(1))[0])
        self._hard_deadline = datetime.fromtimestamp(max(self._y))

    def eval(self, timestamp):
        """ Determines the value of executing task with respect to the given timestamp

        :param timestamp: the input timestamp
        :return: the corresponding value
        """
        if timestamp < min(self._x) or timestamp > max(self._x):    # Out of scheduling window (too early/late)
            return self._invalid_time_value

        # Fixing negative values returned by regression
        return max (self._f(timestamp), self._min_regression_value)

    @abstractmethod
    def name(self) -> str:
        pass

    def utopian_value(self):
        return max(self._y)

    def shift_deadlines(self, shift_interval: float):
        """ Shifts all the deadlines and their associated values

        :param shift_interval: The number of seconds to shift the deadlines
        """
        # Configure new values
        new_values = []
        for deadline, value in self._values:
            new_timestamp = deadline + shift_interval
            new_values.append((new_timestamp, value))

        # Refit model
        self.fit_model(new_values)


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
        #super().__init__()

        # Set's earliest_start/soft_deadline/hard_deadline
        # Note these values aren't used, but are set to avoid issues
        now = datetime.now()
        self._earliest_start: datetime = now
        self._soft_deadline: datetime = now
        self._hard_deadline: datetime = now
        self._value = kwargs.get("CONSTANT_VALUE", 1)

    def fit_model(self, values: List[Tuple[datetime, int]]):
        """ Not Used

        :param values:
        :return:
        """
        pass

    def eval(self, timestamp):
        """ Returns a constant value regardless of the timestamp value

        :param timestamp: a timestamp (not used, value is constant)
        :return: the constant value
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