from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from operator import itemgetter
import numpy as np
import numpy.polynomial.polynomial as poly
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
        self._non_utopian_offset = 1    #
        self._invalid_time_value = 0
        self._earliest_start: datetime = None
        self._soft_deadline: datetime = None
        self._hard_deadline: datetime = None
        self._coef = None
        self._stats = None

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
    def fit_model(self, values: List[Tuple[float, float]]):
        """ Takes a list of datetimes/value tuples and determines the following
        (1) earliest_start
        (2) soft_deadline
        (3) heard_deadline

        Also splits the tuples in to corresponding lists: x = datetimes & y = values

        :param values: a list of execution_time/value tuples
        """
        self._values = values
        self._x, self._y = zip(*values)              # Timestampes, Values
        self._earliest_start = datetime.fromtimestamp(min(self._x))
        self._soft_deadline = datetime.fromtimestamp(max(values, key=itemgetter(1))[0])
        self._hard_deadline = datetime.fromtimestamp(max(self._x))

    def eval(self, timestamp: datetime):
        """ Determines the value of executing task with respect to the given execution_time

        :param timestamp: the input execution_time
        :return: the corresponding value
        """
        if timestamp < min(self._x) or timestamp > max(self._x):    # Out of scheduling window (too early/late)
            return self._invalid_time_value

        value = self._f(timestamp)
        utopian_value = self.utopian_value()
        # Handle case where non-utopian values are equal (or higher) than utopian
        if timestamp != self.soft_deadline and value >= utopian_value:
            value = value - self._non_utopian_offset

        # Fixing negative values returned by regression
        value = max(self._f(timestamp), self._min_regression_value)

        return value

    @abstractmethod
    def name(self) -> str:
        pass

    def utopian_value(self):
        return max(self._y)

    def shift_deadlines(self, shift_interval: timedelta):
        """ Shifts all the deadlines and their associated values

        :param shift_interval: The number of seconds to shift the deadlines
        """
        # Configure new values
        new_values = []
        for deadline, value in self._values:
            shift: float = shift_interval.total_seconds()
            new_timestamp: float = deadline + shift
            new_values.append((new_timestamp, value))

        # Shift Earliest Start and Soft/Hard Deadlines
        self._earliest_start = self._earliest_start + shift_interval
        self._soft_deadline = self._soft_deadline + shift_interval
        self._hard_deadline = self._hard_deadline + shift_interval

        # Refit model
        self.fit_model(new_values)


class NuRegression(AbstractNu):

    def __init__(self):
        """

        """
        super().__init__()

    def fit_model(self, values: List[Tuple[float, int]]):
        """

        :param values:
        :return:
        """
        super().fit_model(values)
        rank = len(self._y) - 1
        xs = list(self._x)
        ys = list(self._y)

        self._f = poly.polyfit(xs, ys, rank) # Generate Polynomial

        #self._coef, self._stats = poly.polyfit(xs, ys, rank)
        #self._f = poly.Polynomial(self._coef)


    def eval(self, timestamp: datetime):
        """

        :param timestamp:
        :return:
        """

        if timestamp < min(self._x) or timestamp > max(self._x):  # Out of scheduling window (too early/late)
            return self._invalid_time_value

        value = poly.polyval(timestamp, self._f) #poly.polyval(timestamp, self._coef)
        utopian_value = self.utopian_value()

        # Handle case where non-utopian values are equal (or higher) than utopian
        if timestamp != self.soft_deadline and value >= utopian_value:
            value = utopian_value - self._non_utopian_offset

        # Fixing negative values returned by regression
        value = max(value, self._min_regression_value)

        assert(value <= self.utopian_value())
        return value

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
        self._value = kwargs.get("CONSTANT_VALUE", 0.5)
        self._values = [(self._earliest_start.timestamp(), self._value)]

    def fit_model(self, values: List[Tuple[datetime, int]]):
        """ Not Used

        :param values:
        :return:
        """
        super().fit_model(values)

    def eval(self, timestamp):
        """ Returns a constant value regardless of the execution_time value

        :param timestamp: a execution_time (not used, value is constant)
        :return: the constant value
        """
        return self._value

    def name(self) -> str:
        return "CONSTANT"

    def utopian_value(self):
        return self._value


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