from abc import ABC, abstractmethod
import random
from time import sleep


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


"""A dummy Analysis Class

"""


class DummyAnalysis(AbstractAnalysis):

    def __init__(self):
        super().__init__()
        self._wcet = 5                      # in Seconds
        self._wcbu = 1                      # in ...

    def execute(self):
        """This function simulates running a dummy taskold.

        :return: void

        The function randoms an execution time close to worst-case execution time (wcet). Note it can exceed it's deadline.

        """
        execution_time = random.randint(self._wcet - 1, self._wcet + 1)
        print("Running Task: " + str(id(self)))
        sleep(execution_time/1000)
        print("Completed Task " + str(id(self)) + " after " + str(execution_time) + " milliseconds")


class AnalysisFactory:

    @staticmethod
    def get_analysis(analysis_type: str):
        analysis = None

        if analysis_type.upper() == "DUMMY":
            analysis = DummyAnalysis()
        else:
            raise Exception("Invalid Analysis Type")

        return analysis