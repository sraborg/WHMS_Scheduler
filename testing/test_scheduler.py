import unittest
from scheduler import *
from task import SleepTask, SystemTask, AbstractTask
from nu import NuFactory
from analysis import MedicalAnalysis


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

    def test_verify_schedule_dependencies_t1(self):
        """ Case: Invalid Schedule (Task 1 is scheduled before task 2 despite depending on task 2).
            Expected: Fails
        """

        schedule = Schedule()
        task_1 = UserTask()
        task_2 = UserTask()
        task_1.add_dependency(task_2)


        schedule.append(task_2)
        schedule.append(task_1)

        # Swap Dependencies to create an invalid schedule
        t = schedule[0]
        schedule[0] = schedule[1]
        schedule[1] = t

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

    def test_verify_schedule_dependencies_t3(self):
        """ Case: Invalid "Schedule" (list of tasks) (Task 1 depends on Tasks 2 but task 2 is not in the list)
            Expected: Pass
        """
        schedule = []
        task_1 = UserTask()
        task_2 = UserTask()
        task_1.add_dependency(task_2)

        schedule.append(task_1)

        self.assertFalse(AbstractScheduler.verify_schedule_dependencies(schedule))

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

    def test_utopian_schedule_2(self):
        """ Verify that the utopian schedule calculates the correct total value (including periodic tasks), regardless
            of time conflicts
            Expected: Pass
        """

        start_time = datetime.fromtimestamp(1640995200)
        end_time = datetime.fromtimestamp(1640997000)

        s = MetaHeuristicScheduler()
        s.start_time = start_time
        s.end_time = end_time
        tasks = UserTask.load_tasks("test_tasks_30_min_horizon.json")
        tasks = Schedule(tasks)

        u_value = s.utopian_schedule_value(tasks)

        aco = SchedulerFactory.ant_colony_scheduler()
        aco.start_time = start_time
        aco.end_time = end_time
        aco_sch = aco.schedule_tasks(tasks, timedelta(minutes=5))
        aco_value = s.simulate_execution(aco_sch)

        elbsa = SchedulerFactory.enhanced_list_based_simulated_annealing()
        elbsa.start_time = start_time
        elbsa.end_time = end_time
        elbsa_sch = elbsa.schedule_tasks(tasks, timedelta(minutes=5))
        elbsa_value = s.simulate_execution(elbsa_sch)

        nga = SchedulerFactory.new_genetic_scheduler()
        nga.start_time = start_time
        nga.end_time = end_time
        nga_sch = nga.schedule_tasks(tasks, timedelta(minutes=5))
        nga_value = s.simulate_execution(nga_sch)

        self.assertEqual(u_value, 300)
        self.assertGreaterEqual(u_value, aco_value)
        self.assertGreaterEqual(u_value, nga_value)
        self.assertGreaterEqual(u_value, elbsa_value)

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

    def test_get_task_details(self):

        start = datetime.now()
        soft_deadline = start + timedelta(minutes=5)
        hard_deadline = start + timedelta(minutes=10)

        #s = Schedule()
        t = UserTask()
        t.values = [
                (start.timestamp(), 1),
                (soft_deadline.timestamp(), 100),
                (hard_deadline.timestamp(), 1)
            ]
        t.analysis = MedicalAnalysis(wcet=5)
        t.nu = NuFactory.regression()
        t.nu.fit_model(t.values)
        #s.append(t)

        execution_time = start + timedelta(minutes=4)

        #sch = MetaHeuristicScheduler()
        #sch.start_time = start
        #sch.end_time = start + timedelta(minutes=15)
        details = t.get_scheduled_task_details(execution_time)

        #print(details)

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

    def test_dependency_sort(self):

        tasks = Schedule()
        for i in range(4):
            t = UserTask()
            if i % 2 == 1:
                t.add_dependency(tasks[i-1])
            tasks.append(t)

        tasks[1].add_dependency(tasks[2])

        s = MetaHeuristicScheduler()
        s1 = s.dependency_sort(tasks)

        tasks.reverse()
        s2 = s.dependency_sort(tasks)


        self.assertTrue(s.verify_schedule_dependencies(s1))
        self.assertTrue(s.verify_schedule_dependencies(s2))


if __name__ == '__main__':
    unittest.main()