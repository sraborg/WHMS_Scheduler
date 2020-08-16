from abc import ABC, abstractmethod


class AbstractAnalysis(ABC):

    def __init__(self):
        self._wcet = None
        self._wcbu = None

    @property
    def wcet(self):
        return self._wcet

    @wcet.setter
    def wcet(self, cost):
        self._wcet = cost

    @property
    def wcbu(self):
        return self._wcbu

    @wcbu.setter
    def wcbu(self, cost):
        self._wcbu = cost

    @abstractmethod
    def execute(self):
        pass