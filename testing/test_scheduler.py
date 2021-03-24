import unittest
from src.scheduler import *
from src.task import SleepTask
from src.nu import NuFactory
from src.analysis import MedicalAnalysis


class TestScheduler(unittest.TestCase):

    def test_no_duplicate_tasks_t1(self):
        """ Case: No duplicates
            Expected: Pass
        """
        tasks = Schedule()
        task_1 = UserTask()
        task_2 = UserTask()
        tasks.append(task_1)
        tasks.append(task_2)

        self.assertTrue(AbstractScheduler._no_duplicate_tasks(tasks))

    def test_no_duplicate_tasks_t2(self):
        """ Case Duplicates present
            Expected: Fail
        """
        tasks = Schedule()
        task_1 = UserTask()
        tasks.append(task_1)
        tasks.append(task_1)

        self.assertFalse(AbstractScheduler._no_duplicate_tasks(tasks))

    def test_no_duplicate_tasks_passes_sleeptasks(self):
        """ Case: SleepTasks with dependencies.
            Expected: Pass (Should ignore SleepTasks)
        """
        tasks = Schedule()
        task_1 = SleepTask()
        task_2 = SleepTask()
        task_1.add_dependency(task_2)
        tasks.append(task_1)
        tasks.append(task_2)

        self.assertTrue(AbstractScheduler._no_duplicate_tasks(tasks))

    def test_verify_schedule_dependencies_t1(self):
        """ Case: Invalid Schedule (Task 1 is scheduled before task 2 despite depending on task 2).
            Expected: Fails
        """

        schedule = Schedule()
        task_1 = UserTask()
        task_2 = UserTask()
        task_1.add_dependency(task_2)

        schedule.append(task_1)
        schedule.append(task_2)

        self.assertFalse(AbstractScheduler.verify_schedule_dependencies(schedule))

    def test_verify_schedule_dependencies_t2(self):
        """ Case: Valid Schedule (Task 1 depends on Tasks 2 and task 2 is scheduled first)
            Expected: Pass
        """
        schedule = Schedule()
        task_1 = UserTask()
        task_2 = UserTask()
        task_1.add_dependency(task_2)

        schedule.append(task_2)
        schedule.append(task_1)

        self.assertTrue(AbstractScheduler.verify_schedule_dependencies(schedule))

    def test_verify_task_dependencies_t1(self):
        """ Case: Task dependencies are in prior_tasks
            Expected: Passes
        """
        prior_tasks = Schedule()
        task_1 = UserTask()
        task_2 = UserTask()
        task_1.add_dependency(task_2)

        prior_tasks.append(task_2)

        self.assertTrue(AbstractScheduler._verify_task_dependencies(task_1, prior_tasks))

    def test_verify_task_dependencies_t2(self):
        """ Case: Task 1 dependencies are not in prior_tasks
            Expected: Fails
        """
        prior_tasks = Schedule()
        task_1 = UserTask()
        task_2 = UserTask()
        task_1.add_dependency(task_2)

        self.assertFalse(AbstractScheduler._verify_task_dependencies(task_1, prior_tasks))

    def test_simulate_execution(self):
        """ Case: Tasks have a total constant value of 3. Simulate_execution should return 3.
            Expected: Pass
        """
        sch = GeneticScheduler()
        schedule = Schedule()
        task_1 = UserTask()
        task_2 = UserTask()
        task_1.nu = NuFactory().get_nu("CONSTANT", CONSTANT_VALUE=1)
        task_1.analysis = MedicalAnalysis(wcet=timedelta(seconds=1))
        task_2.nu = NuFactory().get_nu("CONSTANT", CONSTANT_VALUE=2)
        task_2.analysis = MedicalAnalysis(wcet=timedelta(seconds=1))

        schedule.append(task_1)
        schedule.append(task_2)

        #sch._tasks = schedule

        self.assertEqual(3, sch.simulate_execution(schedule))

    def test_generate_sleep_tasks(self):
        """ Verifies the correct number of sleepTasks are generated.
            Case: 60 minute horizon divided by 1 minute long sleep tasks. Shoul
            Expected: pass
        """
        horizon = timedelta(minutes=60)
        sleep_interval = timedelta(minutes=1)

        sleep_tasks = AbstractScheduler.generate_sleep_tasks(horizon, sleep_interval)
        self.assertEqual(60, len(sleep_tasks))

    def test_generate_periodic_tasks(self):
        """ Verifies the correct number of periodic tasks were created
            Case: task with a 10 second periodicity scheduled over a 1 minute horizon (should generate 5 additional tasks)
            Expected: pass
        """
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=1)

        s = MetaHeuristicScheduler()
        s.start_time = start_time
        s.end_time = end_time
        tasks = Schedule()
        t1 = UserTask()
        t1.periodicity = timedelta(seconds=10)
        t1.nu = NuFactory().get_nu("CONSTANT", CONSTANT_VALUE=1)
        t1.nu._earliest_start = start_time
        t1.nu._soft_deadline = start_time + timedelta(seconds=5)
        t1.nu._hard_deadline = start_time + timedelta(seconds=10)

        tasks.append(t1)

        tasks.extend(s.generate_periodic_tasks(tasks))

        self.assertEqual(len(tasks), 6)

    def test_generate_periodic_tasks_t2(self):
        """ Verifies the last periodic task has correct shifted time
            Case: task with a 10 second periodicity scheduled over a 1 minute horizon (should generate 5 additional tasks)
            Expected: pass
        """
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=1)

        s = MetaHeuristicScheduler()
        s.start_time = start_time
        s.end_time = end_time
        tasks = Schedule()
        t1 = UserTask()
        t1.periodicity = timedelta(seconds=10)
        t1.nu = NuFactory().get_nu("CONSTANT", CONSTANT_VALUE=1)
        t1.nu._earliest_start = start_time
        t1.nu._soft_deadline = start_time + timedelta(seconds=5)
        t1.nu._hard_deadline = start_time + timedelta(seconds=10)

        tasks.append(t1)

        tasks.extend(s.generate_periodic_tasks(tasks))

        first_task_timestamp = int(tasks[0].earliest_start.timestamp())
        last_task_timestamp = int(tasks[5].earliest_start.timestamp())
        self.assertEqual(first_task_timestamp + 50, last_task_timestamp)
    '''
    def test_task_save_and_load_tasklist_pass_match(self):
        """ Test whether save/load tasklist preserves dependencies

        """
        tasklist = []
        t1 = UserTask()
        t2 = UserTask()
        t3 = UserTask()

        t2.add_dependency(t1)
        t3.add_dependency(t1)
        t3.add_dependency(t2)

        tasks = Schedule([t1, t2, t3])
        sys = System()
        sys.scheduler = GeneticScheduler()
        sys.scheduler.save_tasklist("test_save_load.cvs", tasks)
        loaded_tasks = sys.scheduler.load_tasklist("test_save_load.cvs")
        self.assertEqual(sys._tasks, loaded_tasks)
    '''

    def test_fit_to_horizon(self):
        """ Verify fit_to_horizon removes tasks that exceed horizon.
            Case: 5 tasks with 20 second execution times (100 seconds total) with a 90 second planning horizon.
                Fit to horizon should remove 1 task
            Expect: pass
        """
        schedule = Schedule()

        for i in range(5):
            task = UserTask()
            task.analysis = MedicalAnalysis()
            task.analysis.wcet = timedelta(seconds=20)
            schedule.append(task)

        duration = timedelta(seconds=90)

        s = MetaHeuristicScheduler()
        trimmed_schedule = s.fit_to_horizon(schedule, duration)

        self.assertEqual(len(trimmed_schedule), len(schedule)-1)

    def test_generate_random_schedule_t1(self):
        """ Verify generate_random_schedule respects dependencies
            Expected: Pass
        """
        schedule = Schedule()

        # Generate 10 tasks
        # Odd indexed Tasks depend on prior task
        for i in range(10):
            task = UserTask()
            task.analysis = MedicalAnalysis()
            task.analysis.wcet = timedelta(seconds=10)
            if i % 2 == 1:
                task.add_dependency(schedule[i-1])
            schedule.append(task)

        s = MetaHeuristicScheduler()
        s.optimization_horizon = timedelta(seconds=95)
        gen_schedule = s.generate_random_schedule(schedule, timedelta(seconds=1))
        self.assertTrue(s.verify_schedule_dependencies(gen_schedule))

    def test_generate_random_schedule_t2(self):
        """ Verifies the generate_random_schedule creates the correct number of sleepTasks are generated.
                    Case: 60 minute horizon divided by 1 minute long sleep tasks. Shoul
                    Expected: pass
                """

        schedule = Schedule()

        # Generate 2 tasks
        # Odd indexed Tasks depend on prior task
        for i in range(2):
            task = UserTask()
            task.analysis = MedicalAnalysis()
            task.analysis.wcet = timedelta(seconds=10)
            if i % 2 == 1:
                task.add_dependency(schedule[i - 1])
            schedule.append(task)

        s = MetaHeuristicScheduler()
        s.optimization_horizon = timedelta(minutes=60)
        sleep_interval = timedelta(minutes=1)

        random_schedule = s.generate_random_schedule(schedule, sleep_interval)
        num_sleep_tasks = len([s for s in random_schedule if s.is_sleep_task()])

        self.assertEqual(60, num_sleep_tasks)

    def test_utopian_schedule(self):
        """ Verify that the utopian schedule calculates the correct total value (including periodic tasks), regardless
            of time conflicts
            Expected: Pass
        """

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=1)

        s = MetaHeuristicScheduler()
        s.start_time = start_time
        s.end_time = end_time
        tasks = Schedule()
        t1 = UserTask()
        t1.values =[
            (start_time.timestamp(), 1),
            ((start_time + timedelta(seconds=30)).timestamp(), 1),
            (end_time.timestamp(),1)
        ]
        t1.periodicity = timedelta(seconds=10)
        t1.analysis = MedicalAnalysis(wcet=timedelta(seconds=8))
        t1.nu = NuFactory().get_nu("CONSTANT", CONSTANT_VALUE=1)
        t2 = UserTask()
        t2.values = [
            (start_time.timestamp(), 1),
            ((start_time + timedelta(seconds=30)).timestamp(), 1),
            (end_time.timestamp(), 1)
        ]
        t2.periodicity = timedelta(seconds=0)
        t2.analysis = MedicalAnalysis(wcet=55)
        t2.nu = NuFactory().get_nu("CONSTANT", CONSTANT_VALUE=10)
        tasks.append(t1)
        tasks.append(t2)

        u_value = s.utopian_schedule_value(tasks)

        self.assertEqual(u_value, 16)

    def test_weighted_schedule_value(self):
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=1)
        unscheduled_tasks = Schedule()

        for i in range(10):
            t = UserTask()
            t.analysis = MedicalAnalysis(wcet=timedelta(seconds=5))
            t.nu = NuFactory.get_nu("CONSTANT", CONSTANT_VALUE=1)
            t.values = [
                (start_time.timestamp(), 1),
                ((start_time + timedelta(seconds=30)).timestamp(), 1),
                (end_time.timestamp(), 1)
            ]
            unscheduled_tasks.append(t)

        s = MetaHeuristicScheduler()
        s.start_time = start_time
        s.end_time = end_time

        scheduled_tasks = Schedule()
        for i in range(9):
            scheduled_tasks.append(unscheduled_tasks[i])

        raw = s.simulate_execution(scheduled_tasks)
        utopian = s.utopian_schedule_value(unscheduled_tasks)
        weight = s.weighted_schedule_value(unscheduled_tasks, raw)

        self.assertEqual(weight, raw/utopian)


if __name__ == '__main__':
    unittest.main()