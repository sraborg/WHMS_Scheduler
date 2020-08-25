from abc import ABC
from task_builder_interface import TaskBuilderInterface


class AbstractTask(ABC):

    def __init__(self, builder: TaskBuilderInterface):
        self._analysis = builder.analysis
        self._nu = None
        self._deadline = builder.deadline
        self._cost = None
        self._dependent_tasks = builder.dependent_tasks
        self._dynamic_tasks = builder.dynamic_tasks        # potential tasks
        self._future_tasks = []                           # Tasks

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, cost):
        self._cost = cost

    @property
    def analysis(self):
        return self._analysis

    @analysis.setter
    def cost(self, analysis):
        self._analysis = analysis

    @property
    def deadline(self):
        return self._deadline

    @deadline.setter
    def cost(self, deadline):
        self._deadline = deadline

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
        self._analysis.execute()

