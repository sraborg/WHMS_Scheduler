from abstract_task import AbstractTask
from typing import List
from scheduled_task import ScheduledTask
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
        for task in self._tasks:
            task.release_time = self._get_time_in_milliseconds()
            task.execute()
            task.completion_time = self._get_time_in_milliseconds()
            task.execution_time = task.completion_time - task.release_time
        # self._after()

    def _get_time_in_milliseconds(self) -> int:
        return int(round(time.time() * 1000))