from taskold.task_decorator import TaskDecorator
from taskold.abstract_task import AbstractTask


class TaskWithDependencies(TaskDecorator):

    def __init__(self, task: AbstractTask):
        super().__init__(task)

    def execute(self):
        # something with dependenciesa
        self._task.execute()