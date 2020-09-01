from abc import ABC, abstractmethod
from task import AbstractTask, DummyTask, ScheduledTask
from datetime import datetime
from typing import List
from math import ceil
import random


class SchedulerFactory:

    @staticmethod
    def get_scheduler(scheduler_type: str):
        scheduler = None

        if scheduler_type.upper() == "DUMMY":
            scheduler = DummyScheduler()
        else:
            raise Exception("Invalid Analysis Type")

        return scheduler


class AbstractScheduler(ABC):

    def __init__(self):
        pass

    def generate_dummy_tasks(self, tasklist: List[AbstractTask], interval):
        # Calculate the number of dummy tasks needed
        last_interval = max(task.hard_deadline.timestamp() + task.wcet for task in tasklist)
        total_wcet = sum(task.wcet for task in tasklist)
        dif = last_interval + total_wcet - datetime.now().timestamp()
        num_dummy_tasks = ceil(dif / interval)

        # Generated Scheduled Dummy Tasks
        dummy_tasks = []
        for x in range(num_dummy_tasks):
            dummy_tasks.append(ScheduledTask(DummyTask(interval), datetime.now().timestamp()))

        return tasklist + dummy_tasks
    @abstractmethod
    def schedule_tasks(self, tasklist: List[AbstractTask], interval: int) -> List[AbstractTask]:
        pass


class DummyScheduler(AbstractScheduler):

    def __init__(self):
        super().__init__()

    def schedule_tasks(self, tasklist: List[AbstractTask], interval) -> List[AbstractTask]:
        new_tasklist = self.generate_dummy_tasks(tasklist, interval)

        return random.sample(new_tasklist, len(new_tasklist))