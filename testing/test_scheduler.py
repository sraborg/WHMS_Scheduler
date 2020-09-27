import unittest

from src.scheduler import *
from src.task import SleepTask


class TestScheduler(unittest.TestCase):

    def test_no_duplicate_tasks_catches_duplicate_tasks(self):
        tasks = []
        task_1 = DummyTask()
        tasks.append(task_1)
        tasks.append(task_1)

        sch = DummyScheduler()

        self.assertFalse(sch._no_duplicate_tasks(tasks))

    def test_no_duplicate_tasks_ignores_sleeptasks(self):
        tasks = []
        task_1 = SleepTask()
        task_2 = SleepTask()
        tasks.append(task_1)
        tasks.append(task_2)

        sch = DummyScheduler()

        self.assertTrue(sch._no_duplicate_tasks(tasks))

    def test_verify_schedule_dependencies_fails_invalid_schedule(self):
        schedule = []
        task_1 = DummyTask()
        task_2 = DummyTask()
        task_1.add_dependency(task_2)

        schedule.append(task_1)
        schedule.append(task_2)

        sch = DummyScheduler()

        self.assertFalse(sch._verify_schedule_dependencies(schedule))

    def test_verify_schedule_dependencies_passes_valid_schedule(self):
        schedule = []
        task_1 = DummyTask()
        task_2 = DummyTask()
        task_1.add_dependency(task_2)

        schedule.append(task_2)
        schedule.append(task_1)

        sch = DummyScheduler()

        self.assertTrue(sch._verify_schedule_dependencies(schedule))

    def test_verify_task_dependencies_passes_dependencies_met(self):
        prior_tasks = []
        task_1 = DummyTask()
        task_2 = DummyTask()
        task_1.add_dependency(task_2)

        prior_tasks.append(task_2)

        sch = DummyScheduler()

        self.assertTrue(sch._verify_task_dependencies(task_1, prior_tasks))

    def test_verify_task_dependencies_fails_dependencies_not_met(self):
        prior_tasks = []
        task_1 = DummyTask()
        task_2 = DummyTask()
        task_1.add_dependency(task_2)

        sch = DummyScheduler()

        self.assertFalse(sch._verify_task_dependencies(task_1, prior_tasks))
##
    '''
    def test_verify_dependencies_passes_no_dependencies(self):
        prior_tasks = []
        tasks_with_dependencies = []
        task_1 = DummyTask()
        prior_tasks.append(task_1)

        sch = DummyScheduler()

        self.assertTrue(sch._verify_dependencies(tasks_with_dependencies, prior_tasks))

    def test_verify_dependencies_fails_dependencies_with_no_prior_tasks(self):
        prior_tasks = []
        tasks_with_dependencies = []
        task_1 = DummyTask()
        task_2 = DummyTask()
        task_1.add_dependency(task_2)

        tasks_with_dependencies.append(task_1)

        sch = DummyScheduler()

        self.assertFalse(sch._verify_dependencies(tasks_with_dependencies, prior_tasks))
    '''


if __name__ == '__main__':
    unittest.main()