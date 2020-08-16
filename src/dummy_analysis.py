from abstract_analysis import AbstractAnalysis
import random
from time import sleep

class DummyAnalysis(AbstractAnalysis):

    def __init__(self):
        super().__init__()
        self._wcet = 5
        self._wcbu = 1

    def execute(self):
        execution_time = random.randint(self._wcet - 1, self._wcet + 1)
        print("Running Dummy Task")
        sleep(execution_time/1000)
        print("Completed Task after " + str(execution_time) + " milliseconds")
