from abc import ABC, abstractmethod


class AbstractNu(ABC):

    def __init__(self):
        self._x = None              # Time
        self._y = None              # Values
        self._regression = None

    @abstractmethod
    def value(self, time):
        pass


class NuRegression(AbstractNu):

    def __init__(self):
        super().__init__()

    def value(self, time):
        return 1
