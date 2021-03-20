import unittest
from system import System
from datetime import datetime, timedelta
from src.scheduler import *
from src.task import SleepTask
from src.nu import NuFactory, NuConstant
from src.analysis import DummyAnalysis, MedicalAnalysis


class TestScheduler(unittest.TestCase):

    def test_no_duplicate_tasks_pass_no_duplicate_tasklist(self):
        tasks = []
        task_1 = DummyTask()
        task_2 = DummyTask()
        tasks.append(task_1)
        tasks.append(task_2)

        self.assertTrue(AbstractScheduler._no_duplicate_tasks(tasks))

    def test_no_duplicate_tasks_fails_duplicate_tasklist(self):
        tasks = []
        task_1 = DummyTask()
        tasks.append(task_1)
        tasks.append(task_1)

        self.assertFalse(AbstractScheduler._no_duplicate_tasks(tasks))

    def test_no_duplicate_tasks_passes_sleeptasks(self):
        tasks = []
        task_1 = SleepTask()
        task_2 = SleepTask()
        task_1.add_dependency(task_2)
        tasks.append(task_1)
        tasks.append(task_2)

        self.assertTrue(AbstractScheduler._no_duplicate_tasks(tasks))

    def test_verify_schedule_dependencies_fails_invalid_schedule(self):
        schedule = []
        task_1 = DummyTask()
        task_2 = DummyTask()
        task_1.add_dependency(task_2)

        schedule.append(task_1)
        schedule.append(task_2)

        self.assertFalse(AbstractScheduler._verify_schedule_dependencies(schedule))

    def test_verify_schedule_dependencies_passes_valid_schedule(self):
        schedule = []
        task_1 = DummyTask()
        task_2 = DummyTask()
        task_1.add_dependency(task_2)

        schedule.append(task_2)
        schedule.append(task_1)

        self.assertTrue(AbstractScheduler._verify_schedule_dependencies(schedule))

    def test_verify_task_dependencies_passes_dependencies_met(self):
        prior_tasks = []
        task_1 = DummyTask()
        task_2 = DummyTask()
        task_1.add_dependency(task_2)

        prior_tasks.append(task_2)

        self.assertTrue(AbstractScheduler._verify_task_dependencies(task_1, prior_tasks))

    def test_verify_task_dependencies_fails_dependencies_not_met(self):
        prior_tasks = []
        task_1 = DummyTask()
        task_2 = DummyTask()
        task_1.add_dependency(task_2)

        self.assertFalse(AbstractScheduler._verify_task_dependencies(task_1, prior_tasks))

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

    def test_simulate_execution_passes_correct_value(self):
        sch = GeneticScheduler()
        schedule = []
        task_1 = DummyTask()
        task_2 = DummyTask()
        task_1.nu = NuFactory().get_nu("CONSTANT", constant=1)
        task_1._analysis = DummyAnalysis(wcet=1)
        task_2.nu = NuFactory().get_nu("CONSTANT", constant=2)
        task_2._analysis = DummyAnalysis(wcet=1)

        schedule.append(task_1)
        schedule.append(task_2)

        sch._tasks = schedule

        self.assertEqual(3, sch.simulate_execution(schedule))

    def test_simulate_execution_fails_incorrect_value(self):
        sch = GeneticScheduler()
        schedule = []
        task_1 = DummyTask()
        task_2 = DummyTask()
        task_1.nu = NuFactory().get_nu("CONSTANT", constant=1)
        task_1._analysis = DummyAnalysis(wcet=1)
        task_2.nu = NuFactory().get_nu("CONSTANT", constant=2)
        task_2._analysis = DummyAnalysis(wcet=1)

        schedule.append(task_1)
        schedule.append(task_2)

        sch._tasks = schedule

        self.assertNotEqual(10, sch.simulate_execution(schedule))

    def test_calculate_optimization_horizon_passes_correct_value(self):
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

    def test_generate_sleep_tasks_pass_correct_num_sleep_tasks(self):
        start_time = datetime.now() + timedelta(minutes=10)

        schedule = []

        task_1 = DummyTask()
        task_1.hard_deadline = start_time + timedelta(minutes=15)
        task_1._analysis = DummyAnalysis(wcet=60)

        sch = GeneticScheduler()
        schedule.append(task_1)
        sch._tasks = schedule

        sleep_tasks = sch.generate_sleep_tasks(schedule, 60, start_time=start_time)

        self.assertEqual(15, len(sleep_tasks))

    def test_selection_pass_selects_most_fit(self):
        sch = GeneticScheduler()
        sch._breeding_percentage = 0.5

        schedule_1 = []
        schedule_2 = []
        schedule_3 = []
        schedule_4 = []
        task_1 = DummyTask()
        task_2 = DummyTask()
        task_3 = DummyTask()
        task_4 = DummyTask()
        task_5 = DummyTask()
        task_6 = DummyTask()
        task_7 = DummyTask()
        task_8 = DummyTask()
        task_1.nu = NuFactory().get_nu("CONSTANT", constant=1)
        task_1._analysis = DummyAnalysis(wcet=1)
        task_2.nu = NuFactory().get_nu("CONSTANT", constant=23)
        task_2._analysis = DummyAnalysis(wcet=1)
        task_3.nu = NuFactory().get_nu("CONSTANT", constant=93)
        task_3._analysis = DummyAnalysis(wcet=1)
        task_4.nu = NuFactory().get_nu("CONSTANT", constant=15)
        task_4._analysis = DummyAnalysis(wcet=1)
        task_5.nu = NuFactory().get_nu("CONSTANT", constant=2)
        task_5._analysis = DummyAnalysis(wcet=1)
        task_6.nu = NuFactory().get_nu("CONSTANT", constant=3)
        task_6._analysis = DummyAnalysis(wcet=1)
        task_7.nu = NuFactory().get_nu("CONSTANT", constant=6)
        task_7._analysis = DummyAnalysis(wcet=1)
        task_8.nu = NuFactory().get_nu("CONSTANT", constant=28)
        task_8._analysis = DummyAnalysis(wcet=1)

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

    def test_adt_node_choices_pass_preserves_dependencies(self):
        t1 = DummyTask()
        t2 = DummyTask()
        t1.add_dependency(t2)

        ant = Ant()

        tasks = [t1, t2]
        adt = AntDependencyTree(tasks)
        choices = adt.ant_task_choices(ant, 1)
        self.assertTrue(adt.get_ant_task(t1) not in choices)

    def test_ant_visit_pass_correct_path(self):
        t1 = DummyTask()
        t2 = DummyTask()
        t1.add_dependency(t2)

        ant = Ant()

        tasks = [t1, t2]
        adp = AntDependencyTree(tasks)
        t1_node = adp.get_ant_task(t1)
        t1_time = datetime.now().timestamp()
        ant.visit(t1_node, t1_time)
        t2_node = adp.get_ant_task(t2)
        t2_time = datetime.now().timestamp()
        ant.visit(t2_node, t2_time)

        ant_visited_nodes = list(ant.get_visited_nodes())
        prediction = [(t1_node, t1_time), (t2_node, t2_time)]
        self.assertTrue(prediction == ant_visited_nodes)

    def test_ant_last_visite_node_pass(self):
        t1 = DummyTask()
        t2 = DummyTask()
        t1.add_dependency(t2)

        ant = Ant()

        tasks = [t1, t2]
        adp = AntDependencyTree(tasks)
        t1_node = adp.get_ant_task(t1)
        t1_time = datetime.now().timestamp()
        ant.visit(t1_node, t1_time)
        t2_node = adp.get_ant_task(t2)
        t2_time = datetime.now().timestamp()
        ant.visit(t2_node, t2_time)

        last_visited_node = ant.last_visited_node()

        prediction = (t2_node, t2_time)
        self.assertTrue(prediction == last_visited_node)

    def test_adt_node_choices_fail_duplicate(self):
        t1 = DummyTask()
        t2 = DummyTask()
        tasklist = [t1, t2]
        ant = Ant()
        adt = AntDependencyTree(tasklist)

        t1_node = adt.get_ant_task(t1)
        t1_time = datetime.now().timestamp()
        ant.visit(t1_node, t1_time)

        self.assertTrue(t1_node not in adt.ant_task_choices(ant, t1_time + 5))

    def test_adt_node_choices_pass_removes_visited_nodes(self):
        tasks = []
        ant = Ant()

        # Generate 20 Task
        for i in range(20):
            t = DummyTask()
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

    def test_task_save_and_load_tasklist_pass_match(self):
        """ Test whether save/load tasklist preserves dependencies

        """
        tasklist = []
        t1 = DummyTask()
        t2 = DummyTask()
        t3 = DummyTask()

        t2.add_dependency(t1)
        t3.add_dependency(t1)
        t3.add_dependency(t2)

        sys = System()
        sys.add_task(t1)
        sys.add_task(t2)
        sys.add_task(t3)

        sys.scheduler =  GeneticScheduler()
        sys.scheduler.save_tasklist("test_save_load.cvs", sys._tasks)
        loaded_tasks = sys.scheduler.load_tasklist("test_save_load.cvs")
        self.assertEqual(sys._tasks, loaded_tasks)

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

    def test_remove_top_(self):
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
        largest = tq.pop()
        self.assertEqual(largest, 35)

    def test_peek_(self):
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

    def test_PMX(self):

        p1 = UserTask.generate_random_tasks(10)
        p2 = p1[:]
        random.shuffle(p2)

        c1, c2 = MetaHeuristicScheduler.partial_matched_crossover(p1, p2)


        self.assertTrue(MetaHeuristicScheduler._no_duplicate_tasks(c1))
        self.assertTrue(MetaHeuristicScheduler._no_duplicate_tasks(c2))

    def test_generate_sleep_tasks(self):

        schedule = Schedule()
        duration = timedelta(minutes=1)
        s = MetaHeuristicScheduler()
        s.analysis = MedicalAnalysis()
        s.optimization_horizon = timedelta(minutes=1)
        sleep_interval = timedelta(seconds=1)
        sleep_tasks = s.generate_sleep_tasks(s.optimization_horizon, sleep_interval)
        self.assertEqual(len(sleep_tasks), 60)

    def test_fit_to_horizon(self):

        schedule = Schedule()

        for i in range(5):
            task = UserTask()
            task.analysis = MedicalAnalysis()
            task.analysis.wcet = 20
            schedule.append(task)

        duration = timedelta(seconds=90)

        s = MetaHeuristicScheduler()
        trimmed_schedule = s.fit_to_horizon(schedule, duration)

        self.assertEqual(len(trimmed_schedule), len(schedule)-1)

    def test_fit_to_horizon__caching(self):

        schedule = Schedule()

        for i in range(5):
            task = UserTask()
            task.analysis = MedicalAnalysis()
            task.analysis.wcet = 20
            schedule.append(task)

        duration = timedelta(seconds=90)

        s = MetaHeuristicScheduler()
        trimmed_schedule = s.fit_to_horizon(schedule, duration)
        trimmed_schedule2 = s.fit_to_horizon(schedule, duration)

        self.assertTrue(trimmed_schedule is trimmed_schedule2)


    def test_generate_random_schedule(self):

        schedule = Schedule()

        for i in range(10):
            task = UserTask()
            task.analysis = MedicalAnalysis()
            task.analysis.wcet = 10
            if i % 2 == 1:
                task.add_dependency(schedule[i-1])
            schedule.append(task)

        s = MetaHeuristicScheduler()
        s.optimization_horizon = timedelta(seconds=95)
        gen_schedule = s.generate_random_schedule(schedule, timedelta(seconds=1))
        independent_tasks_first = all(list(map(lambda x: x.has_dependencies() == False, gen_schedule[0:5])))
        dependent_tasks_2nd = all(list(map(lambda x: x.has_dependencies(), gen_schedule[5:10])))
        sleep_tasks_last = all(list(map(lambda x: isinstance(x, SleepTask), gen_schedule[10:])))

        self.assertTrue(independent_tasks_first)
        self.assertTrue(dependent_tasks_2nd)
        #self.assertTrue(sleep_tasks_last)

if __name__ == '__main__':
    unittest.main()