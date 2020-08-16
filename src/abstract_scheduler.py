from abc import ABC
from abstract_task import AbstractTask
from typing import Set


class AbstractScheduler(ABC):

    def __init__(self):
        pass

    def generate_schedule(self, tasklist: Set[AbstractTask]) -> Set[AbstractTask]:
        pass
