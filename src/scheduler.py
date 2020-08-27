from abc import ABC, abstractmethod
from task import AbstractTask
from typing import List
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

    @abstractmethod
    def schedule_tasks(self, tasklist: List[AbstractTask]) -> List[AbstractTask]:
        pass


class DummyScheduler(AbstractScheduler):

    def __init__(self):
        super().__init__()

    def schedule_tasks(self, tasklist: List[AbstractTask]) -> List[AbstractTask]:
        return random.sample(tasklist, len(tasklist))