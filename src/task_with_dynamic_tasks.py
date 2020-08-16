from task_decorator import TaskDecorator
from abstract_task import AbstractTask


class TaskWithDynamicTasks(TaskDecorator):

    def __init__(self, task: AbstractTask):
        super().__init__(task)

    def execute(self):
        self._task.execute()
        self._check_dynamic_tasks()

    def _check_dynamic_tasks(self):
        for dyn_task in self.dynamic_tasks:
            if (dyn_task(1)):
                self._task.future_tasks.append(dyn_task(0))

