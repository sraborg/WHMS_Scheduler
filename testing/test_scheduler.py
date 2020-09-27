import unittest

from src.scheduler import *


class TestScheduler(unittest.TestCase):

    def test_no_duplicate_tasks(self):
        tasks = []
        task_1 = DummyTask()
        tasks.append(task_1)
        tasks.append(task_1)

        sch = DummyScheduler()

        self.assertFalse(sch._no_duplicate_tasks(tasks))


if __name__ == '__main__':
    unittest.main()