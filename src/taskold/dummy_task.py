from abstract_task import AbstractTask
from time import sleep


class DummyTask(AbstractTask):

    def __init__(self):
        super().__init__()
        self._runtime = 5

    def run(self):
        sleep(self._runtime)
