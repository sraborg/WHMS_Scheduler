import unittest
from src.scheduler import *
from src.task import SleepTask


class TestTask(unittest.TestCase):

    def test_generate_random_tasks(self):
        start = datetime.now()
        end = start + timedelta(minutes=5)

        unscheduled_tasks = AbstractTask.generate_random_tasks(1,)