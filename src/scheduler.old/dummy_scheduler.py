from typing import List
#from scheduler.old.abstract_scheduler import AbstractScheduler

from task.abstract_task import AbstractTask
from .abstract_scheduler import AbstractScheduler
# from random import sample
import random


class DummyScheduler(AbstractScheduler):

    def __init__(self):
        super().__init__()

    def schedule_tasks(self, tasklist: List[AbstractTask]) -> List[AbstractTask]:
        return random.sample(tasklist, len(tasklist))