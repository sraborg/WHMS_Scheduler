"""A dummy Analysis Class

"""

from taskold.analysis.abstract_analysis import AbstractAnalysis
import random
from time import sleep


class DummyAnalysis(AbstractAnalysis):

    def __init__(self):
        super().__init__()
        self._wcet = 5
        self._wcbu = 1

    def execute(self):
        """This function simulates running a dummy taskold.

        :return: void

        The function randoms an execution time close to worst-case execution time (wcet). Note it can exceed it's deadline.

        """
        execution_time = random.randint(self._wcet - 1, self._wcet + 1)
        print("Running Task: " + str(id(self)))
        sleep(execution_time/1000)
        print("Completed Task " + str(id(self)) + " after " + str(execution_time) + " milliseconds")
