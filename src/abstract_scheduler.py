import abc
from abstract_task import AbstractTask
from typing import Set


class AbstractScheduler:

    def __init__(self):
        pass

    def generateSchedule(self, tasklist: Set[AbstractTask]) -> Set[AbstractTask]:
        pass
