from abstract_task import AbstractTask
from scheduled_task import ScheduledTask
from scheduler_factory import SchedulerFactory
from typing import List
import time


class System:

    def __init__(self):
        self._scheduler = None
        self._tasks: List[ScheduledTask] = []
        self._schedule = []

    def add_task(self, task: AbstractTask):
        queue_time = self._get_time_in_milliseconds()
        self._tasks.append(ScheduledTask(task, queue_time))

    def execute_schedule(self):
        # self._before()
        for task in self._schedule:
            task.release_time = self._get_time_in_milliseconds()
            task.execute()
            task.completion_time = self._get_time_in_milliseconds()
            task.execution_time = task.completion_time - task.release_time
        # self._after()

    def _get_time_in_milliseconds(self) -> int:
        return int(round(time.time() * 1000))

    def set_scheduler(self, name: str):
        self._scheduler = SchedulerFactory.get_scheduler(name)

    def schedule_tasks(self):
       self._schedule = self._scheduler.schedule_tasks(self._tasks)
