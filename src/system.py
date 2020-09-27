from task import AbstractTask, DummyTask#, ScheduledTask
from scheduler import SchedulerFactory
from typing import List
from datetime import datetime
import time


class System:

    def __init__(self, **kwargs):
        self._scheduler = None
        self._tasks = []
        self._schedule = []
        self._interval = .5            # in Seconds
        self._completed_tasks = []

    def add_task(self, task: AbstractTask):
        time = datetime.now()
        #self._tasks.append(ScheduledTask(task, queue_time=time))
        if not task.is_dummy():
            task.queue_time = time
        self._tasks.append(task)

    def execute_schedule(self):
        print("Executing Schedule")
        # self._before()
        total = 0
        for i, task in enumerate(self._schedule):

            # Check for dependencies
            if task.has_dependencies():
                message = "Dependency Error: Attempting to run task " + str(id(task)) + " on iteration " + str(i) + ".\n"
                for dependency in task.get_dependencies():
                    index = self._schedule.index(dependency)
                    message += "Dependent task " + str(id(dependency)) + " scheduled to run on interval " + str(index)
                raise Exception(message)

            task.release_time = datetime.now()
            task.execute()
            task.completion_time = datetime.now()
            #task.execution_time = task.completion_time - task.release_time
            total += task.value()

            #self._schedule.remove(task)

            if not task.is_dummy():
                self._completed_tasks.append(task)

                # Update Dependencies
                for future_task in self._schedule[i:]:
                    future_task.remove_dependency(task)

        # self._after()

        print("\n============================\n Completed " + str(len(self._schedule)) + " tasks for total value of " + str(total))


    def _get_time_in_milliseconds(self) -> int:
        return int(round(time.time() * 1000))

    def set_scheduler(self, name: str):
        self._scheduler = SchedulerFactory.get_scheduler(name)

    def schedule_tasks(self):
        self._schedule = self._scheduler.schedule_tasks(self._tasks, self._interval)

