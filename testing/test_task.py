import unittest
from src.scheduler import *
from src.task import SleepTask


class TestTask(unittest.TestCase):

    def test_generate_random_tasks(self):
        start = datetime.now()
        end = start + timedelta(minutes=5)

        #unscheduled_tasks = AbstractTask.generate_random_tasks(1,)

    def test_adding_dependencies(self):

        t1 = UserTask()
        t2 = UserTask()
        t1.add_dependency(t2)

        self.assertTrue(t2 in t1._dependent_tasks)
        self.assertTrue(t1 in t2._depended_by)

    def test_schedule(self):
        s = Schedule()

        t1 = UserTask()
        t2 = UserTask()
        t1.add_dependency(t2)

        try:
            s.append(t2)
        except RuntimeError:
            pass

        s.append(t1)
