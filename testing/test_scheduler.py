import unittest
from system import System
from datetime import datetime, timedelta
from src.scheduler import *
from src.task import SleepTask
from src.nu import NuFactory
from src.analysis import DummyAnalysis


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


    '''
    def test_adp_visit_node_pass_update_ant_position(self):
        t1 = DummyTask()
        time = datetime.now().timestamp()
        ant = Ant()

        tasks = [t1]
        adp = AntDependencyTree(tasks)
        node = adp.get_node(t1)
        adp.visit_node(ant, node, time)
        self.assertTrue(ant.last_visited_node() == (node, time))

    
    def test_adp_visit_node_pass_update_pheromone(self):
        t1 = DummyTask()
        time = datetime.now().timestamp()
        ant = Ant()
        ant2 = Ant()

        tasks = [t1]
        adp = AntDependencyTree(tasks)
        node = adp.get_node(t1)
        adp.visit_node(ant, node, time)
        adp.visit_node(ant2, node, time)
        edge = adp._pheromones[(None, (node, time))]
        self.assertEqual(edge, 20000)
    '''
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