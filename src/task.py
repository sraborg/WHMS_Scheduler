from abstract_task import AbstractTask
from builder_interface import BuilderInterface


class Task(AbstractTask):

    def __init__(self, builder: BuilderInterface):
        self._analysis = builder._analysis
        self._nu = builder._analysis
        self._deadline = builder._deadline
        #self._cost = None
        self._dependent_tasks = builder._dependent_tasks
        self._dynamic_tasks = builder._dynamic_tasks        # potential tasks
        self._future_tasks = None                           # Tasks

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def dependent_tasks(self, cost):
        self._cost = cost

    @property
    def dependent_tasks(self):
        return self._dependent_tasks

    @dependent_tasks.setter
    def dependent_tasks(self, tasks):
        self._dependent_tasks = tasks

    @property
    def dynamic_tasks(self):
        return self._dynamic_tasks

    @dynamic_tasks.setter
    def dynamic_tasks(self, tasks):
        self._dynamic_tasks = tasks

    @property
    def future_tasks(self):
        return self._future_tasks

    @future_tasks.setter
    def future_tasks(self, tasks):
        self._future_tasks = tasks

    def execute(self):
        pass

