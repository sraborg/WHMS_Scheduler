from task import AbstractTask, ScheduledTask
from scheduler import SchedulerFactory
from typing import List
from datetime import datetime
import time


class System:

    def __init__(self):
        self._scheduler = None
        self._tasks: List[ScheduledTask] = []
        self._schedule = []
        self._interval = 5000

    def add_task(self, task: AbstractTask):
        queue_time = datetime.now()
        self._tasks.append(ScheduledTask(task, queue_time))

    def execute_schedule(self):
        # self._before()
        total = 0
        for task in self._schedule:
            task.release_time = datetime.now()
            task.execute()
            task.completion_time = datetime.now()
            task.execution_time = task.completion_time - task.release_time
            total += task.value()
            print("taskold value: " + str(task.value()))
        # self._after()

        print("Completed " + str(len(self._schedule)) + " tasks for total value of " + str(total))

    def _get_time_in_milliseconds(self) -> int:
        return int(round(time.time() * 1000))

    def set_scheduler(self, name: str):
        self._scheduler = SchedulerFactory.get_scheduler(name)

    def schedule_tasks(self):
       self._schedule = self._scheduler.schedule_tasks(self._tasks)

