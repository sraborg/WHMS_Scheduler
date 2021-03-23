import unittest
from scheduler import GeneticScheduler
from task import UserTask, Schedule
from datetime import datetime, timedelta


class TestGeneticSchedulers(unittest.TestCase):

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

    def test_PMX_(self):
        """ Test PMX with a known solution
            Expected: pass
        """
        sch = GeneticScheduler()
        p1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        p2 = [9, 3, 7, 8, 2, 6, 5, 1, 4]

        c1, c2 = sch.partial_matched_crossover(p1, p2, 3, 7)

        self.assertEqual(c1, [9, 3, 2, 4, 5, 6, 7, 1, 8])

    def test_PMX_t1(self):
        """ Verify method throws errors when crossover indices are invalid
        """
        sch = GeneticScheduler()
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
        sch = GeneticScheduler()
        p1 = [1, 'a', 4, 3]
        p2 = [3, 1, 2, 4]

        c1, c2 = sch.partial_matched_crossover(p1, p2, 1, 3)

        self.assertEqual(c1, [3, 'a', 4, 2])
