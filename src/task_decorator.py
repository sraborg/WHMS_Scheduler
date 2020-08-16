from abc import ABC
from abstract_task import AbstractTask


class TaskDecorator(ABC):

    def __init__(self, task: AbstractTask):
        super().__init__()
        self._task = task

    @property
    def cost(self):
        return self._task.cost

    @cost.setter
    def cost(self, cost):
        self._task.cost = cost

    @property
    def analysis(self):
        return self._task.analysis

    @analysis.setter
    def deadline(self, analysis_type):
        self._task.analysis = analysis_type

    @property
    def deadline(self):
        return self._task.deadline

    @deadline.setter
    def deadline(self, deadline):
        self._task.deadline = deadline

    @property
    def dependent_tasks(self):
        return self._task.dependent_tasks

    @dependent_tasks.setter
    def dependent_tasks(self, tasks):
        self._task.dependent_tasks = tasks

    @property
    def dynamic_tasks(self):
        return self._task.dynamic_tasks

    @dynamic_tasks.setter
    def dynamic_tasks(self, tasks):
        self._task.dynamic_tasks = tasks

    @property
    def future_tasks(self):
        return self._task.future_tasks

    @future_tasks.setter
    def future_tasks(self, tasks):
        self._task.future_tasks = tasks

    def execute(self):
        self._task.execute()
