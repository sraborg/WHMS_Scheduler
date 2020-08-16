from abc import ABC, abstractmethod
from abstract_task import AbstractTask
from typing import List


class AbstractScheduler(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def schedule_tasks(self, tasklist: List[AbstractTask]) -> List[AbstractTask]:
        pass
