import unittest
from scheduler import Ant, AntDependencyTree
from task import UserTask, Schedule
from datetime import datetime, timedelta


class TestAntSchedulers(unittest.TestCase):

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
            if i % 2 == 0:  # Ant visits even tasks
                t = adt.get_ant_task(task)
                ant.visit(t, datetime.now().timestamp())

        choices = adt.ant_task_choices(ant, 60)

        # Fail should be empty
        fail = [visited_task for visited_task in ant.get_completed_ant_tasks() if visited_task in choices]
        self.assertTrue(len(fail) == 0)

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

