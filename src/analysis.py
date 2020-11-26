from abc import ABC, abstractmethod
import random
from time import sleep


class AbstractAnalysis(ABC):

    def __init__(self, **kwargs):
        self.wcet = kwargs.get("wcet", 1)
        self.wcbu = kwargs.get("wcbu", 1)

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def name(self) -> str:
        pass


"""A dummy Analysis Class

"""


class DummyAnalysis(AbstractAnalysis):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.precision = kwargs.get("precision", (0, 0))

    def execute(self):
        """This function simulates running a dummy taskold.

        :return: void

        The function randoms an execution time close to worst-case execution time (wcet). Note it can exceed it's deadline.

        """
        lower_bound = self.wcet - self.precision[0]
        upper_bound = self.wcet + self.precision[1]

        execution_time = random.uniform(lower_bound, upper_bound)
        print("Running Task: " + str(id(self)))
        sleep(execution_time)
        print("Completed Task " + str(id(self)) + " after " + str(execution_time) + " seconds")

    def name(self):
        return "DUMMY"

class SleepAnalysis(AbstractAnalysis):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def execute(self):
        """This function simulates running a dummy taskold.

        :return: void

        The function randoms an execution time close to worst-case execution time (wcet). Note it can exceed it's deadline.

        """
        print("Waiting... ")
        sleep(self.wcet)

    def name(self):
        return "SLEEP"


class AnalysisFactory:

    @staticmethod
    def get_analysis(analysis_type: str):
        analysis = None

        if analysis_type.upper() == "DUMMY":
            analysis = DummyAnalysis()
        else:
            raise Exception("Invalid Analysis Type")

        return analysis

    @classmethod
    def dummy_analysis(cls):
        return DummyAnalysis()

    @classmethod
    def sleep_analysis(cls):
        return SleepAnalysis()
