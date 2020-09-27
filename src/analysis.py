from abc import ABC, abstractmethod
import random
from time import sleep


class AbstractAnalysis(ABC):

    def __init__(self, **kwargs):
        if "wcet" in kwargs:
            self._wcet = kwargs.get("wcet")
        else:
            self._wcet = 1
        if "wcbu" in kwargs:
            self._wcbu = kwargs.get("wcbu")
        else:
            self._wcbu = 1

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._precision = (0, 0)

    def execute(self):
        """This function simulates running a dummy taskold.

        :return: void

        The function randoms an execution time close to worst-case execution time (wcet). Note it can exceed it's deadline.

        """
        lower_bound = self._wcet - self._precision[0]
        upper_bound = self._wcet + self._precision[1]

        execution_time = random.uniform(lower_bound, upper_bound)
        print("Running Task: " + str(id(self)))
        sleep(execution_time)
        print("Completed Task " + str(id(self)) + " after " + str(execution_time) + " seconds")


class SleepAnalysis(AbstractAnalysis):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def execute(self):
        """This function simulates running a dummy taskold.

        :return: void

        The function randoms an execution time close to worst-case execution time (wcet). Note it can exceed it's deadline.

        """
        print("Waiting... ")
        sleep(self._wcet)


class AnalysisFactory:

    @staticmethod
    def get_analysis(analysis_type: str):
        analysis = None

        if analysis_type.upper() == "DUMMY":
            analysis = DummyAnalysis()
        else:
            raise Exception("Invalid Analysis Type")

        return analysis


