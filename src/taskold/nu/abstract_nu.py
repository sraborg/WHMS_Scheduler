from abc import ABC, abstractmethod


class AbstractNu(ABC):

    def __init__(self):
        self._x = None              # Time
        self._y = None              # Values
        self._regression = None

    @abstractmethod
    def value(self, time):
        pass
