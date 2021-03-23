import unittest
from system import System
from datetime import datetime, timedelta
from src.scheduler import *
from src.task import SleepTask
from src.nu import NuFactory, NuConstant
from src.analysis import DummyAnalysis, MedicalAnalysis


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

    def test_no_duplicate_tasks_fails_duplicate_tasklist(self):
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

    """
    def test_all_tasks_present_fails_missing(self):
        original_schedule = []
        new_schedule = []
        task_1 = DummyTask()
        task_2 = DummyTask()
        original_schedule.append(task_1)
        original_schedule.append(task_2)
        new_schedule.append(task_1)
    
        self.assertFalse(GeneticScheduler._all_tasks_present(original_schedule, new_schedule))
    
    
    def test_all_tasks_present_passes_all_present(self):
        original_schedule = []
        new_schedule = []
        task_1 = DummyTask()
        task_2 = DummyTask()
        original_schedule.append(task_1)
        original_schedule.append(task_2)
        new_schedule.append(task_1)
        new_schedule.append(task_2)

        self.assertTrue(GeneticScheduler._all_tasks_present(original_schedule, new_schedule))
    """

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

    def test_calculate_optimization_horizon(self):
        start_time = datetime.now() + timedelta(minutes=10)
        schedule = []

        task_1 = DummyTask()
        task_2 = DummyTask()
        task_1.hard_deadline = start_time + timedelta(minutes=15)
        task_1._analysis = DummyAnalysis(wcet=5)
        task_2.hard_deadline = start_time + timedelta(minutes=5)
        task_2._analysis = DummyAnalysis(wcet=4)

        schedule.append(task_1)
        schedule.append(task_2)

        expected_answer = (start_time + timedelta(minutes=15, seconds=5)).timestamp()

        self.assertEqual(AbstractScheduler.calculate_optimization_horizon(schedule), expected_answer)

    def test_generate_sleep_tasks(self):
        """ Verifies the correct number of sleepTasks are generated.
            Case: 60 minute horizon divided by 1 minute long sleep tasks. Shoul
            Expected: pass
        """
        horizon = timedelta(minutes=60)
        sleep_interval = timedelta(minutes=1)

        sleep_tasks = AbstractScheduler.generate_sleep_tasks(horizon, sleep_interval)
        self.assertEqual(60, len(sleep_tasks))

    """
    def test_selection_pass_selects_most_fit(self):
        sch = GeneticScheduler()
        sch._breeding_percentage = 0.5

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=1)

        sch.start_time = start_time
        sch.end_time = end_time

        schedule_1 = []
        schedule_2 = []
        schedule_3 = []
        schedule_4 = []
        task_1 = UserTask()
        task_2 = UserTask()
        task_3 = UserTask()
        task_4 = UserTask()
        task_5 = UserTask()
        task_6 = UserTask()
        task_7 = UserTask()
        task_8 = UserTask()
        task_1.nu = NuFactory().get_nu("CONSTANT", CONSTANT_VALUE=1)
        task_1.analysis = DummyAnalysis(wcet=timedelta(seconds=1))
        task_2.nu = NuFactory().get_nu("CONSTANT", CONSTANT_VALUE=23)
        task_2.analysis = DummyAnalysis(wcet=timedelta(seconds=1))
        task_3.nu = NuFactory().get_nu("CONSTANT", CONSTANT_VALUE=93)
        task_3.analysis = DummyAnalysis(wcet=timedelta(seconds=1))
        task_4.nu = NuFactory().get_nu("CONSTANT", CONSTANT_VALUE=15)
        task_4.analysis = DummyAnalysis(wcet=timedelta(seconds=1))
        task_5.nu = NuFactory().get_nu("CONSTANT", CONSTANT_VALUE=2)
        task_5.analysis = DummyAnalysis(wcet=timedelta(seconds=1))
        task_6.nu = NuFactory().get_nu("CONSTANT", CONSTANT_VALUE=3)
        task_6.analysis = DummyAnalysis(wcet=timedelta(seconds=1))
        task_7.nu = NuFactory().get_nu("CONSTANT", CONSTANT_VALUE=6)
        task_7.analysis = DummyAnalysis(wcet=timedelta(seconds=1))
        task_8.nu = NuFactory().get_nu("CONSTANT", CONSTANT_VALUE=28)
        task_8.analysis = DummyAnalysis(wcet=timedelta(seconds=1))

        # value = 132
        schedule_1.append(task_1)
        schedule_1.append(task_2)
        schedule_1.append(task_3)
        schedule_1.append(task_4)
        value_1 = sch.simulate_execution(schedule_1)

        # value = 39
        schedule_2.append(task_5)
        schedule_2.append(task_6)
        schedule_2.append(task_7)
        schedule_2.append(task_8)
        value_2 = sch.simulate_execution(schedule_2)

        # value = 113
        schedule_3.append(task_3)
        schedule_3.append(task_4)
        schedule_3.append(task_5)
        schedule_3.append(task_6)
        value_3 = sch.simulate_execution(schedule_3)

        # value = 58
        schedule_4.append(task_7)
        schedule_4.append(task_8)
        schedule_4.append(task_1)
        schedule_4.append(task_2)
        value_4 = sch.simulate_execution(schedule_4)

        sch._tasks = schedule_1
        population = [schedule_1, schedule_2]

        predicted_answer = [schedule_1]
        most_fit = sch._selection(population)
        self.assertTrue(predicted_answer == most_fit)
    """

    def test_adt_node_choices_pass_preserves_dependencies(self):
        """ Verifies that the ADT::ant_task_choices does not include any tasks who's dependencies haven't been met
            Expected: pass
        """
        t1 = UserTask()
        t2 = UserTask()
        t1.add_dependency(t2)

        ant = Ant()

        tasks = [t1, t2]
        adt = AntDependencyTree(tasks)
        choices = adt.ant_task_choices(ant, 1)
        self.assertTrue(adt.get_ant_task(t1) not in choices)

    def test_ant_visit_pass_correct_path(self):
        """ Verify Ant::get_visited_nodes correctly tracks the nodes visited.
            Case: manually have an ant visit two nodes and check method
            Expected: Pass
        """

        t1 = UserTask()
        t2 = UserTask()
        t1.add_dependency(t2)

        ant = Ant()

        tasks = Schedule([t1, t2])
        adp = AntDependencyTree(tasks)
        t1_node = adp.get_ant_task(t1)
        t1_time = datetime.now()
        ant.visit(t1_node, t1_time)
        t2_node = adp.get_ant_task(t2)
        t2_time = datetime.now()
        ant.visit(t2_node, t2_time)

        ant_visited_nodes = list(ant.get_visited_nodes())
        prediction = [(t1_node, t1_time), (t2_node, t2_time)]
        self.assertTrue(prediction == ant_visited_nodes)

    def test_ant_last_visited_node(self):
        """ Verify Ant::last_visited_node returns the correct value.
            Case: Manually have an ant visit a node and verify method
            Expected: pass
        """
        t1 = UserTask()
        t2 = UserTask()
        t1.add_dependency(t2)

        ant = Ant()

        tasks = Schedule([t1, t2])
        adp = AntDependencyTree(tasks)
        t1_node = adp.get_ant_task(t1)
        t1_time = datetime.now()
        ant.visit(t1_node, t1_time)
        t2_node = adp.get_ant_task(t2)
        t2_time = datetime.now()
        ant.visit(t2_node, t2_time)

        last_visited_node = ant.last_visited_node()

        prediction = (t2_node, t2_time)
        self.assertTrue(prediction == last_visited_node)

    def test_adt_node_choices_fail_duplicate(self):
        t1 = UserTask()
        t2 = UserTask()
        tasklist = Schedule([t1, t2])
        ant = Ant()
        adt = AntDependencyTree(tasklist)

        t1_node = adt.get_ant_task(t1)
        t1_time = datetime.now()
        ant.visit(t1_node, t1_time)

        self.assertTrue(t1_node not in adt.ant_task_choices(ant, t1_time + timedelta(seconds=5)))

    def test_adt_node_choices_pass_removes_visited_nodes(self):
        tasks = []
        ant = Ant()

        # Generate 20 Task
        for i in range(20):
            t = UserTask()
            tasks.append(t)

        adt = AntDependencyTree(tasks)

        # Have ant visit first 10 AntTasks
        for i, task in enumerate(tasks):
            if i % 2 == 0:          # Ant visits even tasks
                t = adt.get_ant_task(task)
                ant.visit(t, datetime.now().timestamp())


        choices = adt.ant_task_choices(ant, 60)

        # Fail should be empty
        fail = [visited_task for visited_task in ant.get_completed_ant_tasks() if visited_task in choices]
        self.assertTrue(len(fail) == 0)

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

    def test_permutate_schedule_by_blockinsert_preserves_list_size(self):
        """
        Tests if the Metaheuristic Scheduler method "permute_by_block_insert" preserves the list size.
        :return: Returns True if the size remains the same, otherwise false
        """
        l = [1,2,3,4,5,6,7,8,9,0]
        l2 = l[:]
        scheduler = MetaHeuristicScheduler()
        MetaHeuristicScheduler.permute_schedule_by_blockinsert(l)

        self.assertEqual(len(l), len(l2))

    def test_permutate_schedule_by_reverse_preserves_list_size(self):
        """
        Tests if the Metaheuristic Scheduler method "permute_by_block_reverse" preserves the list size.
        :return: Returns True if the size remains the same, otherwise false
        """
        l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        l2 = l[:]
        scheduler = MetaHeuristicScheduler()
        MetaHeuristicScheduler.permute_schedule_by_inverse(l)

        self.assertEqual(len(l), len(l2))

    def test_permutate_schedule_by_transport_preserves_list_size(self):
        """
        Tests if the Metaheuristic Scheduler method "permute_by_block_reverse" preserves the list size.
        :return: Returns True if the size remains the same, otherwise false
        """
        l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        l2 = l[:]
        scheduler = MetaHeuristicScheduler()
        MetaHeuristicScheduler.permute_schedule_by_transport(l)

        self.assertEqual(len(l), len(l2))

    def test_remove_top_1(self):
        """ Verify TemperatureQueue::remove_top actually removes the largest temperature.
            Case: Largest temperature is 35.
            Expected: Pass
        """
        tq = TemperatureQueue()

        tq.push(19)
        tq.push(32)
        tq.push(35)
        tq.push(23)

        largest = tq.pop()
        self.assertEqual(largest, 35)

    def test_remove_top_(self):
        """ Verify TemperatureQueue::remove_top removes the correct number of temperatures.
            Case: Remove 2 items and verify queue removes 2 items
            Expected: Pass
        """
        tq = TemperatureQueue()

        tq.push(19)
        tq.push(32)
        tq.push(35)
        tq.push(23)
        tq.push(82)
        tq.push(49)

        before = len(tq)
        tq.remove_top(2)
        after = len(tq)
        self.assertEqual(before, after + 2)

    def test_peek_(self):
        """ Verify TemperatureQueue::peak returns largest temperature without removing it
            Case: 82 is the largest temperature and 49 is the second largest.
            Expected: Pass
        """
        tq = TemperatureQueue()

        tq.push(19)
        tq.push(32)
        tq.push(35)
        tq.push(23)
        tq.push(82)
        tq.push(49)

        p1 = tq.peek()
        self.assertEqual(p1, 82)
        p2 = tq.peek()
        self.assertEqual(p1, p2)

    def test_generate_random_schedulea(self):
        tasklist = []

        for i in range(20):
            tasklist.append(UserTask())

        for i in range(5):
            task = UserTask()
            d1 = random.choice(tasklist)
            d2 = random.choice(tasklist)

            task.add_dependency(d1)
            task.add_dependency(d2)
            tasklist.append(task)

        for task in tasklist:
            task.analysis = MedicalAnalysis()
            task.analysis.wcet = 20
            task.nu = NuConstant()
            task.nu.value = 1


        start = datetime.now()

        tasklist[0].periodicity = 600

        sch = MetaHeuristicScheduler()
        sch.start_time = start
        sch.end_time = start + timedelta(minutes=30)

        schedule = sch.generate_random_schedule(tasklist, 60)
        self.assertTrue(sch._validate_schedule(schedule))

    def test_PMX_(self):
        """ Test PMX with a known solution
            Expected: pass
        """
        sch = MetaHeuristicScheduler()
        p1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        p2 = [9, 3, 7, 8, 2, 6, 5, 1, 4]

        c1, c2 = sch.partial_matched_crossover(p1, p2, 3, 7)

        self.assertEqual(c1, [9, 3, 2, 4, 5, 6, 7, 1, 8])

    def test_PMX_t1(self):
        """ Verify method throws errors when crossover indices are invalid
        """
        sch = MetaHeuristicScheduler()
        p1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        p2 = [9, 3, 7, 8, 2, 6, 5, 1, 4]

        try:
            c1, c2 = sch.partial_matched_crossover(p1, p2, 0, 7)  # x1 too small
        except IndexError as msg:
            print("Caught IndexError: " + str(msg))

        try:
            c1, c2 = sch.partial_matched_crossover(p1, p2, 7, 7)  # x1 too large
        except IndexError as msg:
            print("Caught IndexError: " + str(msg))

        try:
            c1, c2 = sch.partial_matched_crossover(p1, p2, 1, 1)  # x2 too small
        except IndexError as msg:
            print("Caught IndexError: " + str(msg))

        try:
            c1, c2 = sch.partial_matched_crossover(p1, p2, 6, 8)  # x1 too large
        except IndexError as msg:
            print("Caught IndexError: " + str(msg))

    def test_PMX_t2(self):
        """ Test PMX with a known solution
                    Expected: pass
        """
        sch = MetaHeuristicScheduler()
        p1 = [1, 'a', 4, 3]
        p2 = [3, 1, 2, 4]

        c1, c2 = sch.partial_matched_crossover(p1, p2, 1, 3)

        self.assertEqual(c1, [3, 'a', 4, 2])

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


if __name__ == '__main__':
    unittest.main()