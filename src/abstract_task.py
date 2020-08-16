from abc import ABC
from builder_interface import BuilderInterface


class AbstractTask(ABC):

    def __init__(self, builder: BuilderInterface):
        self._analysis = builder.analysis
        self._nu = None
        self._deadline = builder.deadline
        #self._cost = None
        self._dependent_tasks = builder.dependent_tasks
        self._dynamic_tasks = builder.dynamic_tasks        # potential tasks
        self._future_tasks = None                           # Tasks

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, cost):
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

