from abc import ABC, abstractmethod
from operator import itemgetter
import numpy as np
from task import AbstractTask, SleepTask, AntTask, UserTask, Schedule
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from math import ceil, floor, log, exp
import sympy
import random
import copy
import heapq
from functools import reduce


class SchedulerFactory:

    @staticmethod
    def get_scheduler(scheduler_type: str):
        scheduler = None

        if scheduler_type.upper() == "GENETIC":
            scheduler = GeneticScheduler()
        elif scheduler_type.upper() == "ANT":
            scheduler = AntScheduler()
        elif scheduler_type.upper() == "RANDOM":
            scheduler = RandomScheduler()
        else:
            raise Exception("Invalid Analysis Type")

        return scheduler

    @classmethod
    def simulated_annealing(cls,
                            temperature=10000,
                            max_iterations=100,
                            threshold=0.01,
                            generational_threshold=10,
                            start_time=None,
                            end_time=None,
                            verbose=False,
                            invalid_schedule_value=-1000.0,
                            **kwargs):

        if start_time is None:
            start_time = datetime.now()

        return SimulateAnnealingScheduler(
                            temperature=temperature,
                            max_iterations=max_iterations,
                            threshold=threshold,
                            generational_threshold=generational_threshold,
                            start_time=start_time,
                            end_time=end_time,
                            verbose=verbose,
                            invalid_schedule_value=invalid_schedule_value,
                            **kwargs)

    @classmethod
    def enhanced_list_based_simulated_annealing(cls,
                                                temperature=10000,
                                                max_iterations=100,
                                                threshold=0.01,
                                                generational_threshold=10,
                                                start_time=None,
                                                end_time=None,
                                                verbose=False,
                                                invalid_schedule_value=-1000.0,
                                                **kwargs):

        if start_time is None:
            start_time = datetime.now()

        return EnhancedListBasedSimulatedAnnealingScheduler(
            temperature=temperature,
            max_iterations=max_iterations,
            threshold=threshold,
            generational_threshold=generational_threshold,
            start_time=start_time,
            end_time= end_time,
            verbose=verbose,
            invalid_schedule_value=invalid_schedule_value,
            **kwargs)

    @classmethod
    def ant_scheduler(cls,
                      colony_size=25,
                      alpha=1,
                      beta=1,
                      epsilon=0.5,
                      max_iterations=100,
                      threshold=0.01,
                      generational_threshold=10,
                      start_time=None,
                      verbose=False,
                      invalid_schedule_value=-1000.0,
                      **kwargs):

        if start_time is None:
            start_time = datetime.now()
        return AntScheduler(
            colony_size=colony_size,
            alpha=alpha,
            beta= beta,
            epsilon= epsilon,
            max_iterations=max_iterations,
            threshold=threshold,
            generational_threshold=generational_threshold,
            start_time=start_time,
            verbose=verbose,
            invalid_schedule_value=invalid_schedule_value,
            **kwargs)

    @classmethod
    def ant_colony_scheduler(cls,
                      colony_size=25,
                      alpha=1,
                      beta=1,
                      epsilon=0.5,
                      max_iterations=100,
                      threshold=0.01,
                      generational_threshold=10,
                      start_time=None,
                      verbose=False,
                      invalid_schedule_value=-1000.0,
                      **kwargs):

        if start_time is None:
            start_time = datetime.now()
        return AntColonyScheduler(
            colony_size=colony_size,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            max_iterations=max_iterations,
            threshold=threshold,
            generational_threshold=generational_threshold,
            start_time=start_time,
            verbose=verbose,
            invalid_schedule_value=invalid_schedule_value,
            **kwargs)

    @classmethod
    def ElitistAntScheduler(cls,
                             colony_size=25,
                             alpha=1,
                             beta=1,
                             epsilon=0.5,
                             max_iterations=100,
                             threshold=0.01,
                             generational_threshold=10,
                             start_time=None,
                             verbose=False,
                             invalid_schedule_value=-1000.0,
                             **kwargs):

        if start_time is None:
            start_time = datetime.now()
        return ElitistAntScheduler(
            colony_size=colony_size,
            alpha=alpha,
            beta=beta,
            epsilon=epsilon,
            max_iterations=max_iterations,
            threshold=threshold,
            generational_threshold=generational_threshold,
            start_time=start_time,
            verbose=verbose,
            invalid_schedule_value=invalid_schedule_value,
            **kwargs)

    @classmethod
    def genetic_scheduler(cls,
                          population_size=5000,
                          breeding_percentage=0.05,
                          mutation_rate=0.01,
                          max_iterations=100,
                          threshold=0.01,
                          generational_threshold=10,
                          start_time=None,
                          verbose=False,
                          invalid_schedule_value=-1000.0,
                          elitism=True,
                          **kwargs):
        """

        :param population_size:
        :param breeding_percentage:
        :param mutation_rate:
        :param max_iterations:
        :param threshold:
        :param generational_threshold:
        :param start_time:
        :param verbose:
        :param invalid_schedule_value:
        :param elitism:
        :param kwargs:
        :return:
        """
        if start_time is None:
            start_time = datetime.now()

        return GeneticScheduler(
            population_size=population_size,
            breeding_percentage=breeding_percentage,
            mutation_rate=mutation_rate,
            max_iterations=max_iterations,
            threshold=threshold,
            generational_threshold=generational_threshold,
            start_time=start_time,
            verbose=verbose,
            invalid_schedule_value=invalid_schedule_value,
            elitism=elitism,
            **kwargs)

    @classmethod
    def new_genetic_scheduler(cls,
                          population_size=5000,
                          breeding_percentage=0.05,
                          mutation_rate=0.01,
                          max_iterations=100,
                          threshold=0.01,
                          generational_threshold=10,
                          start_time=None,
                          verbose=False,
                          invalid_schedule_value=-1000.0,
                          elitism=True,
                          **kwargs):
        """

        :param population_size:
        :param breeding_percentage:
        :param mutation_rate:
        :param max_iterations:
        :param threshold:
        :param generational_threshold:
        :param start_time:
        :param verbose:
        :param invalid_schedule_value:
        :param elitism:
        :param kwargs:
        :return:
        """
        if start_time is None:
            start_time = datetime.now()

        return NewGeneticScheduler(
            population_size=population_size,
            breeding_percentage=breeding_percentage,
            mutation_rate=mutation_rate,
            max_iterations=max_iterations,
            threshold=threshold,
            generational_threshold=generational_threshold,
            start_time=start_time,
            verbose=verbose,
            invalid_schedule_value=invalid_schedule_value,
            elitism=elitism,
            **kwargs)

    @classmethod
    def random_scheduler(cls,
                         sample_size=1000,
                         start_time = None,
                         **kwargs):

        if start_time is None:
            start_time = datetime.now()

        return RandomScheduler(
            sample_size=sample_size,
            start_time=start_time,
            **kwargs)


class AbstractScheduler(ABC):

    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        self._tasks = None
        self._optimization_horizon = None
        self.start_time: datetime = kwargs.get("start_time", datetime.now())
        self.end_time: datetime = kwargs.get("end_time", None)
        self.verbose = kwargs.get("verbose", False)
        self.invalid_schedule_value = kwargs.get("invalid_schedule_value", -1000.0)
        self._utopian_schedule_value = None
        self._flag_generate_sleep_tasks = True
        self._flag_generate_periodic_tasks = True

        # Cache
        self._objective_cache = {}
        self._trimmed_schedule_cache = {}
        self._schedule_values_cache: Dict[float] = {}
        self._utopian_schedule_value_cache: {}

    @property
    def optimization_horizon(self):
        """ Accessor method for the optimization_horizon attribute.

        Note if the scheduler's end_time attribute is not set, an optimization_horizon will be calculated (based off the tasklist)
        and the end_time will be set.

        :return:
        """

        # Lazy-Load
        if self._optimization_horizon is None:

            if self.end_time is None:
                raise ValueError("End Time attribute must be set")
            if self.start_time is None:
                raise ValueError("Start Time attribute must be set")

            self._optimization_horizon = self.end_time - self.start_time
            if self._optimization_horizon.total_seconds() < 0:
                raise RuntimeError("Invalid Horizon. End-time must be after Start-time")
        return self._optimization_horizon


    @optimization_horizon.setter
    def optimization_horizon(self, value):
        self._optimization_horizon = value

    """
    def calculate_optimization_horizon(self, tasklist):
        
        temp = [task.hard_deadline for task in tasklist]
        latest_task = max([task.hard_deadline.timestamp() + task.wcet for task in tasklist])
        horizon = datetime.fromtimestamp(latest_task) - self.start_time
        return horizon

    """

    def _initialize_tasklist(self, tasklist: List[AbstractTask], interval):
        """ Template Method that calls several delegated tasks

        Calls appropriate delegated functions to add sleep/periodic tasks

        :param tasklist: the list of tasks
        :param interval: sleep time in seconds
        :return: a (potentially) modified tasklist
        """
        self._tasks = new_tasklist = tasklist

        if self.end_time is None or self._optimization_horizon is None:
            self._optimization_horizon = self.optimization_horizon          # Somewhat redundant

        if self._flag_generate_periodic_tasks:
            new_tasklist = tasklist + self.generate_periodic_tasks(tasklist)

        if self._flag_generate_sleep_tasks:
            new_tasklist = new_tasklist + self.generate_sleep_tasks(tasklist, interval)

        return new_tasklist

    @staticmethod
    def generate_sleep_tasks(planning_horizon: timedelta, sleep_interval: timedelta): #tasklist: List[AbstractTask], interval):
        """ Generates sleep tasks for the task list. It works by first calculating the maximum execution
        of entire task list (using each tasks WCET). Then it fills the remaining time left within the
        planning horizon with sleep tasks.

        :param tasklist: The list of tasks
        :param interval:
        :return: A new task list with sleep tasks added
        """

        num_sleep_tasks = floor(planning_horizon / sleep_interval)

        # Generated Scheduled Sleep Tasks
        sleep_tasks = Schedule()
        for x in range(num_sleep_tasks):
            sleep_tasks.append(SleepTask(wcet=sleep_interval))

        return sleep_tasks

    def generate_periodic_tasks(self, tasklist: Schedule):
        """  Generates all possible periodic tasks that can be executed within planning horizon

        :param tasklist: A list of tasks
        :return: A new list with the periodic tasks added
        """

        if len(tasklist.generated_periodic_tasks) > 0:
            raise RuntimeError("Periodic Tasks Have already been generated")

        new_periodic_tasks = Schedule()

        for task in tasklist.periodic_tasks:
            #num_periodic_tasks: int = 0

            i = 1
            t = task.earliest_start
            while t < self.end_time - task.periodicity:

                # interval shift
                shift: timedelta = timedelta(seconds=(i) * task.periodicity.total_seconds())

                new_task: AbstractTask = copy.deepcopy(task)
                new_task.periodicity = timedelta(seconds=-1)

                new_task.nu.shift_deadlines(shift)

                new_periodic_tasks.append(new_task)

                i = i + 1
                t = task.earliest_start + shift

        return new_periodic_tasks

    def clear_cache(self):
        """ Deletes caches for:
            1) Trimmed Schedules
            2) Objective Values
            3) Schedule values
        """

        self._objective_cache.clear()
        self._trimmed_schedule_cache.clear()
        self._schedule_values_cache.clear()

    def _validate_schedule(self, schedule: List[AbstractTask], planning_horizon: timedelta) -> bool:
        """ Checks if the (trimmed) schedule is consistent with dependencies
            (e.g. no task is scheduled before any of its dependencies).
            Note: Tasks that exceed the planning horizon are validated

        :param schedule:
        :param planning_horizon:
        :return:
        """

        t_schedule = self.fit_to_horizon(schedule, planning_horizon)

        # Check for Duplicates
        if not AbstractScheduler._no_duplicate_tasks(t_schedule):
            return False

        # Check Dependencies
        if not AbstractScheduler.verify_schedule_dependencies(t_schedule):
            return False

        return True

    @staticmethod
    def verify_schedule_dependencies(schedule: Schedule):
        """ Checks that every dependency for every task is scheduled prior to the task

        :param schedule:
        :return:
        """
        #non_sleep_tasks = [task for task in schedule if not task.is_sleep_task()]
        prior_tasks = []

        # Check Each scheduledTask
        for i, task in enumerate(schedule):

            if task.is_sleep_task():
                continue

            # Check Dependencies
            if not AbstractScheduler._verify_task_dependencies(task, prior_tasks):
                return False

            prior_tasks.append(task)

        return True

    @staticmethod
    def _verify_task_dependencies(task, prior_tasks):
        """ Checks that every dependency for a task is scheduled prior to the task.
        Used internally by verify_schedule_dependencies method.

        :param task:
        :param prior_tasks:
        :return:
        """
        if task.has_dependencies():
            dependencies = task.get_dependencies()

            for dependency in dependencies:
                if dependency not in prior_tasks:
                    return False
        return True

    def load_tasklist(self, filename) -> List[AbstractTask]:
        """Delegated Function

        :param filename: the name of the file to load
        :return: the loaded tasklist
        """
        return AbstractTask.load_tasks(filename)

    def save_tasklist(self, filename, tasklist):
        """

        :param filename: the name of the file to save
        :param tasklist: the tasklist to save
        """
        AbstractTask.save_tasks(filename, tasklist)

    @staticmethod
    def _no_duplicate_tasks(tasklist: Schedule) -> bool:
        """ Makes sure that there are no copies of an exact same Task

        :param tasklist: The list of tasks
        :return: True/False
        """
        for task in tasklist:

            # ingore sleepTasks
            if task.is_sleep_task():
                continue

            if tasklist.count(task) > 1:
                return False

        return True

    @abstractmethod
    def schedule_tasks(self, tasklist: List[AbstractTask], interval: int) -> List[AbstractTask]:
        """

        :param tasklist:
        :param interval:
        :return:
        """
        pass

    def simulate_execution(self, tasklist: List[AbstractTask], start=None, **kwargs):
        """Simulates executes of a schedule. Assumes the schedule is valid.

        :param tasklist:
        :param start:
        :param kwargs:
        :return: the schedule's value
        """
        if start is None:
            time = self.start_time
        else:
            time = start

        #if not AbstractScheduler._validate_schedule(tasklist):
            #return self.invalid_schedule_value

        key = (tuple(tasklist), time)
        if key in self._schedule_values_cache.keys():
            return self._schedule_values_cache[key]
        else:

            total_value = 0

            for task in tasklist:

                total_value += task.value(timestamp=time)
                time += task.wcet

            self._schedule_values_cache[key] = total_value

        #if self._schedule_values_cache[key] > self.utopian_schedule_value(tasklist):
        #    print(str(self._schedule_values_cache[key]) + " | " + str(self.utopian_schedule_value(tasklist)))
        #    raise AssertionError("Current Value exceeds Utopian Value")
        return self._schedule_values_cache[key]

    def utopian_schedule_value(self, schedule: Schedule):
        """ Gets the Utopian (the best) value. Note this value may not be achievable.

        :param schedule:
        :return:
        """
        u_schedule = schedule[:]

        # Generate Periodic Tasks if they haven't been yet
        if len(schedule.generated_periodic_tasks) == 0:
            u_schedule.extend(self.generate_periodic_tasks(schedule))

        value = 0
        for task in u_schedule:
            date_time = task.soft_deadline
            time = date_time
            value += task.value(timestamp=time)

        return value

    def weighted_schedule_value(self, schedule, value=None):
        """

        :param schedule:
        :param value:
        :return:
        """
        if not schedule:
            raise ValueError("Schedule cannot be empty")

        if value is None:
            value = self.simulate_execution(schedule, datetime.now().timestamp())

        utopian_value = self.utopian_schedule_value(schedule)
        weighted_value = value / utopian_value

        return weighted_value

    def generate_random_schedule(self, schedule: Schedule, sleep_interval: timedelta):
        """

        :param tasklist:
        :param interval:
        :return:
        """
        new_schedule = Schedule(schedule.independent_tasks)
        new_schedule.extend(Schedule(schedule.dependent_tasks))

        if self._no_duplicate_tasks(new_schedule) is False:
            print(new_schedule)

        new_schedule.extend(self.generate_periodic_tasks(new_schedule))


        # Handle Sleep Tasks
        sleep_tasks = self.generate_sleep_tasks(self.optimization_horizon, sleep_interval)

        # Insert Sleep Tasks into random positions
        for sleep_task in sleep_tasks:
            index = random.randint(0, len(new_schedule)-1)
            new_schedule.insert(index, sleep_task)

        return new_schedule

    @staticmethod
    def calulate_total_schedule_execution_time(schedule):
        total = sum([t.wcet for t in schedule])
        return total

    def fit_to_horizon(self, schedule, planning_horizon: timedelta, cache=True):
        """ Returns a schedule that fits within the planning horizon.

        If the schedule exceeds the horizon, the excess tasks are truncated.
        Schedules that are within the horizon are unaltered


        :param schedule:
        :param planning_horizon:
        :param cache Boolean to enable caching
        :return: a schedule that fits the horizon
        """

        key = None
        if cache:
            s = tuple(schedule)
            key = (s, planning_horizon)
            if key in self._trimmed_schedule_cache:
                return self._trimmed_schedule_cache[key]

        trimmed_schedule = Schedule()

        exec_time = timedelta(0)
        for task in schedule:

            # If running the next task exceeds the horizon, stop
            if exec_time + task.wcet > planning_horizon:
                break

            exec_time = exec_time + task.wcet
            trimmed_schedule.append(task)

        if cache:
            self._trimmed_schedule_cache[key] = trimmed_schedule
        return trimmed_schedule

    def _objective(self, schedule):
        """ Gives a quantifable value to a trimmed schedule.

        :param schedule: The schedule to evaluate
        :return: a value
        """
        horizon = self.optimization_horizon

        t_schedule = self.fit_to_horizon(schedule, horizon)

        key = tuple(t_schedule)
        if key in self._objective_cache:
            return self._objective_cache[key]

        if not self._validate_schedule(t_schedule, horizon):
            self._objective_cache[key] = self.invalid_schedule_value
        else:
            self._objective_cache[key] = self.simulate_execution(t_schedule)

        return self._objective_cache[key]


class MetaHeuristicScheduler(AbstractScheduler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.learning_duration = timedelta(minutes=kwargs.get("learning_duration", 1)) # Duration Stored in minutes
        self.max_iterations = kwargs.get("max_iterations", 100)
        self.threshold = kwargs.get("threshold", 0.01)
        self.generational_threshold = kwargs.get("generational_threshold", 10)
        self._generational_threshold_count = 0
        self._last_value: float = 0
        self._current_value: float = 0
        self._progress = 0

        self._converged = False
        self._flag_termination_by_duration = True
        self._flag_termination_by_max_iteration = False
        self._flag_termination_by_generational_delta = False

    def _termination_by_duration(self, start_runtime: datetime) -> bool:
        """ Checks whether an the algorithm has exceeded it's allowed duration.

        :param start_runtime: The time the algorithm started
        :return: True/False
        """
        elapsed_time = datetime.now() - start_runtime

        progress: int = ceil((elapsed_time.total_seconds() / self.learning_duration.total_seconds() * 100))
        if self._progress != progress:
            self._progress = progress
            if self.verbose:
                print(str(progress)+"%")
            elif progress % 10 == 0:
                print(str(progress) + "%")
        if progress >= 100:
            return True

        return False

    def _termination_by_max_iterations(self, iteration: int) -> bool:
        """ Checks whether the algorithm has reached its max iteration

        :param iteration: current iteration
        :return: True/False
        """
        if iteration >= self.max_iterations:
            return True
        else:
            return False

    def _termination_by_generational_delta(self, last_value: float, current_value: float):
        """

        :param last_value:
        :param current_value:
        :return:
        """
        delta = abs(last_value - current_value)

        if delta < self.threshold:
            self._generational_threshold_count += 1

            if self._generational_threshold_count >= self.generational_threshold:
                return True
        else:
            self._generational_threshold_count = 0

        if self.verbose:
            print("Delta: " + str(delta) + " | threshold_count: " + str(self._generational_threshold_count))
        return False

    def max_iterations_reached(self):
        pass

    def schedule_tasks(self, tasklist: List[AbstractTask], interval: int) -> List[AbstractTask]:
        pass

    @staticmethod
    def _all_tasks_present(original_schedule, new_schedule):
        for task in original_schedule:
            if task not in new_schedule:
                return False

        return True

    @staticmethod
    def permute_schdule_by_swap(schedule: List, first_index=None):
        """
        Permutes a given schedule in place by swapping the positions of two random tasks.
        User may optionally choose one of the tasks to swap by passing in its index.

        :param schedule: The original Schedule
        :param first_index: (Optional) the index of the a task to be swapped.
        """

        if schedule is None or len(schedule) < 2:
            raise ValueError("Schedule must have at least 2 tasks.")

        l = len(schedule)-1
        if first_index is None:
            first_index = random.randint(0, l)
        second_index = random.randint(0, l)

        temp = schedule[first_index]
        schedule[first_index] = schedule[second_index]
        schedule[second_index] = temp

    @staticmethod
    def permute_schedule_by_blockinsert(schedule: List, first_index=None):
        """ Permutes a given schedule in place by:
         (1) selecting two indices,
         (2) taking the segment starting with the index further in the list to the end of the list,
         (3) shifting the segment to right after the first index

         User may optionally choose one of the tasks to swap by passing in its index.

         Ex) Given schedule [1,2,3,4,5] and indices (0,3) --> [1,4,5,2,3]

        :param schedule: The original Schedule
        :param first_index: (Optional) the index of the a task to use.
        """
        if schedule is None or len(schedule) < 2:
            raise ValueError("Schedule must have at least 2 tasks.")

        l = len(schedule) - 1
        if first_index is None:
            first_index = random.randint(0, l)
        second_index = random.randint(0, l)

        # Make sure first_index is smallest
        if second_index < first_index:
            temp = first_index
            first_index = second_index
            second_index = temp

        block = schedule[second_index:]
        del schedule[second_index:]
        schedule[first_index+1:first_index+1] = block

    @staticmethod
    def permute_schedule_by_inverse(schedule: List, first_index=None):
        """ Permutes a given schedule in place by:
         (1) selecting two indices,
         (2) reversing the order of the segment between the two indices,

         User may optionally choose one of the tasks to swap by passing in its index.

         Ex) Given schedule [1,2,3,4,5] and indices (0,3) --> [4,3,2,1,5]

        :param schedule: The original Schedule
        :param first_index: (Optional) the index of the a task to use.
        """

        if schedule is None or len(schedule) < 2:
            raise ValueError("Schedule must have at least 2 tasks.")

        l = len(schedule) - 1
        if first_index is None:
            first_index = random.randint(0, l)
        second_index = random.randint(0, l)

        # Make sure first_index is smallest
        if second_index < first_index:
            temp = first_index
            first_index = second_index
            second_index = temp

        block = schedule[first_index+1:second_index]
        block.reverse()
        schedule[first_index + 1:second_index] = block

    @staticmethod
    def permute_schedule_by_transport(schedule: List, first_index=None):
        """ Permutes a given schedule in place by:
         (1) selecting two indices,
         (2) taking the segment between the two indices,
         (3) placing the segment into a random point in the schedule

         User may optionally choose one of the tasks to use by passing in its index.

         Ex) Given schedule [1,2,3,4,5] and indices (2,3) (and a random insertion index of 1) --> [1,3,4,2,5]

        :param schedule: The original Schedule
        :param first_index: (Optional) the index of the a task to use.
        """

        if schedule is None or len(schedule) < 2:
            raise ValueError("Schedule must have at least 2 tasks.")

        l = len(schedule) - 1
        if first_index is None:
            first_index = random.randint(0, l)
        second_index = random.randint(0, l)

        # Make sure first_index is smallest
        if second_index < first_index:
            temp = first_index
            first_index = second_index
            second_index = temp

        block = schedule[first_index:second_index]
        del schedule[first_index:second_index]

        insertion_index = random.randint(0, len(schedule) - 1)
        schedule[insertion_index:insertion_index] = block

    def partial_matched_crossover(self, parent_1: List, parent_2, x1=None, x2=None):
        """ Performs the PMX genetic operation on parent_1 and parent_2 to produce child_1 and child_2
            Assumes that neither parent has duplicate elements

        :param parent_1: First Parent
        :param parent_2: Second Parent
        :param x1: First crossover point
        :param x2: Second crossover point
        :return: child_1 and child_2
        """

        # Get Crossover Points
        if x1 is None:
            x1 = random.randint(1, len(parent_1)-2)
        if x2 is None:
            x2 = random.randint(x1+1, len(parent_2)-1)

        if x1 < 1 or x1 > len(parent_1)-2:
            raise IndexError("x1 is not a valid index")
        if x2 <= x1 or x2 > len(parent_2)-1:
            raise IndexError("x2 is not a valid index")

        child_1 = self._get_pmx_child(parent_1, parent_2, x1, x2)
        child_2 = self._get_pmx_child(parent_2, parent_1, x1, x2)

        return child_1, child_2

    def _get_pmx_map(self, e, p1: List, p2: List, x1: int, x2: int, lookup: list):
        """ Recursively find the index that value of element e should map to.
            Used internally by _get_pmx_child.

        :param e: elment
        :param p1: parent 1
        :param p2: parent 2
        :param x1: crossover point 1
        :param x2: crossover point 2
        :param lookup: lookup list for recursion
        :return: the index
        """
        i = p2.index(e)
        v = p1[i]
        j = p2.index(v)

        if p2[j] in lookup:
            l2 = lookup[:]
            l2.remove(p2[i])
            return self._get_pmx_map(p2[j], p1, p2, x1, x2, l2)
        else:
            return j

    def _get_pmx_child(self, p1, p2, x1, x2):
        """ Generates a child from two parents using the modified PMX genetic operation.


        :param p1: parent 1
        :param p2: parent 2
        :param x1: First crossover point
        :param x2: Second crossover point
        :return: child
        """
        s1 = p1[x1:x2]
        s2 = p2[x1:x2]

        c = [None for i in range(len(p1))]

        c[x1:x2] = s1

        unique = [e for e in s2 if e not in s1]

        for e in unique:

            if p1[p2.index(e)] in p2:
                try:
                    mapped_index = self._get_pmx_map(e, p1, p2, x1, x2, s2)
                except ValueError as msg:
                    pass
                else:
                    c[mapped_index] = e

        for i in range(len(c)):
            if c[i] is None:
                c[i] = p2[i]

        return c

    def get_average_schedule_value_from_population(self, population: List[List[AbstractTask]]):
        """

        :param population:
        :return:
        """
        total_value = sum(list(map(lambda x: self._objective(x), population)))
        average = total_value / len(population)
        return average

    def get_max_schedule_value_from_population(self, population: List[List[AbstractTask]]):
        """

        :param population:
        :return:
        """
        return max([self._objective(i) for i in population])


class RandomScheduler(AbstractScheduler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sample_size = kwargs.get("sample_size", 1000)

    def schedule_tasks(self, tasklist: List[AbstractTask], interval) -> List[AbstractTask]:
        print("Generating Schedule with Random Scheduler")
        new_task_list = self._initialize_tasklist(tasklist, interval)

        # Generate possible schedules
        best_solution = None
        best_solution_value = 0

        for i in range(self.sample_size):
            current_solution = random.sample(new_task_list, len(new_task_list))
            current_solution_value = self.simulate_execution(current_solution)
            if current_solution_value > best_solution_value:
                best_solution = current_solution
                best_solution_value = current_solution_value

        return best_solution


class GeneticScheduler(MetaHeuristicScheduler):
    """

    """
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super().__init__(**kwargs)
        self.population_size = kwargs.get("population_size", 5000)
        self.breeding_percentage = kwargs.get("breeding_percentage", 0.05)
        self.mutation_rate = kwargs.get("mutation_rate", 0.01)
        self.elitism = kwargs.get("elitism", True)
        self.max_generations = self.max_iterations
        self._tasks = None

    def schedule_tasks(self, tasklist: List[AbstractTask], interval) -> List[AbstractTask]:
        """

        :param tasklist:
        :param interval:
        :return:
        """
        print("Generating Schedule Using Genetic Algorithm")
        start_runtime = datetime.now()
        new_task_list = self._initialize_tasklist(tasklist, interval)
        population = []
        i = 1
        converged = False

        # Initialize Population
        for x in range(self.population_size):
            population.append(random.sample(new_task_list, len(new_task_list)))

        current_best_schedule_value = 0

        while not converged:
            if self.verbose:
                print("Processing Generation " + str(i))

            breeding_sample = self._selection(population)
            new_best_schedule_value = self._fitness(breeding_sample[0])
            if self.verbose:
                print("Best Fit: " + str(new_best_schedule_value))
            next_generation = self._crossover(breeding_sample, len(population))
            next_generation = self._mutation(next_generation)

            # Generate next Generation
            if self.elitism:
                population = [*breeding_sample, *next_generation]
            else:
                population = next_generation

            # Termination by Duration
            if self._flag_termination_by_duration and self._termination_by_duration(start_runtime):
                break

            # Termination by Max Iteration
            if self._flag_termination_by_max_iteration and self._termination_by_max_iterations(i):
                print("Max iterations met" + str(i) + " | " + str(self.max_iterations))
                break

            # Termination by Generational Delta
            if self._flag_termination_by_generational_delta and \
                    self._termination_by_generational_delta(current_best_schedule_value, new_best_schedule_value):
                print("Generational Delta Threshold Met")
                print("Convergence Met after " + str(i) + " iterations")
                converged = True

            # Prepare for next iteration
            current_best_schedule_value = new_best_schedule_value
            i += 1

        best_fit = self._selection(population)[0]

        if self.verbose:
            if converged:
                print("Convergence Meet after " + str(i) + " iterations")
            else:
                print("Convergence not meet. Algorithm quit after " + str(i) + " iterations")

        return best_fit

    def _fitness(self, schedule, tasklist=None):
        """

        :param schedule:
        :param tasklist:
        :return:
        """
        if tasklist is None:
            tasklist = self._tasks

        if not AbstractScheduler._validate_schedule(schedule, self.optimization_horizon):
            return self.invalid_schedule_value
        #elif not GeneticScheduler._all_tasks_present(tasklist, schedule):
            #return self.invalid_schedule_value
        else:
            return self.simulate_execution(schedule)

    def _selection(self, population, **kwargs):
        """

        :param population:
        :param kwargs:
        :return:
        """
        values = []
        for schedule in population:
            values.append(self._fitness(schedule))

        sample = list(zip(population, values))
        sample.sort(key=lambda item: item[1], reverse=True)
        ordered_sample, _ = zip(*sample)
        cutoff = ceil(len(sample)*self.breeding_percentage)
        parent_sample = list(ordered_sample[:cutoff])

        return parent_sample

    def _crossover(self, parents, population_size):
        """ Creates the next generation of schedules. If "elitism is set", parents are carried over to next generation.

        :param parents:
        :param population_size:
        :return:
        """
        next_generation = []

        if self.elitism:
            next_gen_size = population_size - len(parents)
        else:
            next_gen_size = population_size

        for x in range(next_gen_size):
            p1, p2 = random.sample(parents, 2)
            crossover_point = random.randint(0, population_size)
            child = p1[:crossover_point] + p2[crossover_point:]
            next_generation.append(child)

        return next_generation

    def _mutation(self, population):
        """ Probabilistically "mutates" each task in the schedule based on the mutation rate.
        When a mutation occurs the task position is swapped with another task in the schedule (at random)

        :param population: The population (schedule) to mutate
        :return: returns the mutated schedule
        """
        for schedule in population:
            for i, task in enumerate(schedule):
                if random.random() <= self.mutation_rate:
                    self.permute_schdule_by_swap(schedule, i)

        return population


class NewGeneticScheduler(MetaHeuristicScheduler):
    """

    """
    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super().__init__(**kwargs)
        self.num_populations = kwargs.get("num_populations", 10)
        self.population_size = kwargs.get("population_size", 10)
        self.breeding_percentage = kwargs.get("breeding_percentage", 0.5)

    def schedule_tasks(self, tasklist: Schedule, sleep_interval) -> List[AbstractTask]:
        """ Generates a schedule for a list of tasks

        :param tasklist: a list of tasks
        :param interval:
        :return: a schedule
        """

        self.clear_cache()

        print("Generating Schedule Using " + self.algorithm_name() + " Algorithm")
        start_runtime = datetime.now()

        # Generation initial populations
        populations = []
        for i in range(self.num_populations):
            populations.append([])
            for chromosome in range(self.population_size):
                c = self.generate_random_schedule(tasklist, sleep_interval)
                #print(self._fitness(c))
                populations[i].append(c)

        i = 1
        current_best_schedule = []
        current_best_schedule_value = float('-inf')

        while True:
            if self.verbose:
                print("Processing Generation " + str(i))

            new_populations = []
            for population in populations:

                # Selection
                breeding_sample = self._selection(population)

                # Crossover
                children = self._crossover(population, breeding_sample)

                # Mutation
                self._mutation(population, children)

                # Configure Next generation
                if len(children) != 0:
                    next_generation = population[:-len(children)] + children
                else: # Handle fringe case where no children were created
                    next_generation = population

                new_populations.append(next_generation)

                # Termination innerloop by Duration
                if self._flag_termination_by_duration and self._termination_by_duration(start_runtime):
                    print("Learning Duration Ended. Retrieving Best Schedule Found")
                    break

            # Sort Each Chromosome in Each Populations by fitness
            for population in new_populations:
                population.sort(key=self._fitness, reverse=True)

            # update best schedule
            iterative_best_schedules = list(map(lambda x: x[0], new_populations))

            iterative_best_schedule = max(iterative_best_schedules, key= lambda x: self._fitness(x))
            if self._fitness(iterative_best_schedule) > current_best_schedule_value:
                current_best_schedule = iterative_best_schedule


            # Migration
            #for i in range(int(len(new_populations)/2)):
            #    self._migrate(new_populations[i], new_populations[len(new_populations)-1-i])

            # Update Population
            populations = new_populations

            # Termination by Duration
            if self._flag_termination_by_duration and self._termination_by_duration(start_runtime):
                break

        return self.fit_to_horizon(current_best_schedule, self.optimization_horizon)

    def _fitness(self, schedule):
        """ Alias for objective method

        :param schedule: a schedule to evaluate
        :return: returns a value
        """

        return self._objective(schedule)

    def _selection(self, population):
        """ Selects a subset of the population for breeding

        :param population: The population to select from
        :return: The breeding sample
        """
        population.sort(key= lambda x: self._fitness(x), reverse=True)
        cutoff = ceil(len(population) * self.breeding_percentage)
        return population[:cutoff]

    def _crossover(self, population, breeding_sample):
        """ Creates the next generation of schedules. If "elitism is set", parents are carried over to next generation.

        :param parents:
        :param population_size:
        :return:
        """
        children = []

        for individual in breeding_sample:

            crossover_rate = self._get_adaptive_crossover_rate(population, individual)

            if random.uniform(0,1) < crossover_rate:
                mate = random.choice(breeding_sample)
                family = []

                c1, c2 = self.partial_matched_crossover(individual, mate)

                #crossover_point = random.randint(0, len(individual))
                #child = individual[:crossover_point] + mate[crossover_point:]
                family.append(individual)
                family.append(mate)
                family.append(c1)
                family.append(c2)

                family.sort(key= lambda x : self._fitness(x), reverse=True)

                for i in range(2):
                    children.append(family[i])

        return children

    def _mutation(self, population, children):
        """ Probabilistically "mutates" each task in the schedule in place based on the mutation rate.
        When a mutation occurs the task position is swapped with another task in the schedule (at random)

        :param population: The population (schedule) to mutate
        """
        for child in children:
            mutation_rate = self._get_adaptive_mutation_rate(population, child)

            #for i, task in enumerate(child):
            if random.random() < mutation_rate:
                i = random.randint(0, len(population))
                test = child[:]
                self.permute_schdule_by_swap(test, i)

                # Only allow mutation if it results in a valid schedule
                if self._validate_schedule(test, self.optimization_horizon):
                    self.permute_schdule_by_swap(child, i)

    def algorithm_name(self):
        return "New Genetic Algorithm"

    def _get_adaptive_crossover_rate(self, population, individual):
        """ Adaptively generates a crossover rate

        :param population: The population the individual comes from
        :param individual: The individual (schedule)
        :return: a crossover rate
        """

        max_value = self.get_max_schedule_value_from_population(population)
        average = self.get_average_schedule_value_from_population(population)
        value = self._fitness(individual)
        if value >= average:

            # Catch fringe case where all individuals have the same value
            if max_value - average == 0:
                return 0
            else:
                return (max_value - value)/(max_value - average)
        else:
            return 1

    def _get_adaptive_mutation_rate(self, population, individual):
        """ Adaptively generates a mutation rate

        :param population: The population the individual comes from
        :param individual: The individual (schedule)
        :return: a mutation rate
        """

        max_value = self.get_max_schedule_value_from_population(population)
        average = self.get_average_schedule_value_from_population(population)
        value = self._fitness(individual)
        if value >= average:
            return 0.5 * (max_value - value)/(max_value -average)
        else:
            return 0.5

    '''
    def _evolutionary_rate(self, population):
        total_value = sum(list(map(lambda x: self.simulate_execution(x), population)))
        average = total_value / len(population)
        best = self.simulate_execution(population[0])
        return (best - average)/average

    def _segregate(self, population):
        """

        :param population: a population whose individuals are sorted in decending order by their fitness value
        :return:
        """
        total_value = sum(list(map(lambda x: self.simulate_execution(x), population)))
        average = total_value / len(population)

        midpoint = 1
        for individual in population:
            if self.simulate_execution(individual) > average:
                midpoint = midpoint+1
            else:
                break

        upper = population[:midpoint+1]
        lower = population[midpoint+1:]

        return lower, upper

    def _migrate(self, population_1, population_2):
        l1, h1 = self._segregate(population_1)
        l2, h2 = self._segregate(population_2)

        population_1 = l1 + h2
        population_2 = l2 + h1

    '''


class AntScheduler(MetaHeuristicScheduler):

    ANT_SYSTEM = 0
    ELITIST_ANT_SYSTEM = 1
    ANT_COLONY = 2
    RANKED_BASED_ANT_SYSTEM = 3
    ITERATIVE_BEST = 0
    BEST_SO_FAR = 1

    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super().__init__(**kwargs)
        self.colony_size = kwargs.get("colony_size", 15)
        self.alpha = kwargs.get("alpha", 1)
        self.beta = kwargs.get("beta", 1)
        self.epsilon = kwargs.get("epsilon", 0.5)
        self.pheromone_update_method = self.RANKED_BASED_ANT_SYSTEM
        self.best_method = kwargs.get("best_method", self.BEST_SO_FAR)
        self.rank = kwargs.get("rank", 5)
        self._flag_generate_sleep_tasks = False
        self._flag_aco_pheromone_update = True
        self._flag_local_pheromone_update = False

        if self.rank > self.colony_size:
            self.rank = ceil(self.colony_size/2)

    def schedule_tasks(self, tasklist: Schedule, interval: timedelta) -> List[AbstractTask]:
        """ Generates a schedule using a modified ant colony optimization algorithm

        :param tasklist: the list of tasks to be scheduled
        :param interval: the period of time to divide the optimization horizon
        :return: an ordered list of tasks
        """
        print("Generating Schedule Using " + self.algorithm_name() + " Algorithm")
        start_runtime = datetime.now()
        new_task_list = self._initialize_tasklist(tasklist, interval)

        adt = AntDependencyTree(new_task_list, min_max=True)

        #root_node = AntTask(SleepTask(wcet=timedelta(seconds=0)))

        # Get Valid starting nodes
        possible_starting_task = adt.ant_task_choices(Ant(), interval)
        root_node = random.choice(possible_starting_task)
        i = 1
        converged = False
        self._generational_threshold_count = 0

        global_solutions: List[Ant] = [] #[([], 0)]  # List of (Schedule, Value)
        current_best_schedule_value = 0

        while not converged:
            if self.verbose:
                print("Processing Swarm " + str(i) + " of " + str(self.colony_size) + " ants")

            # Generate Ant Swarm
            colony = []
            for ant in range(self.colony_size):
                colony.append(Ant())

            # Send each Ant to explore
            for ant in colony:

                # Place ant on starting node
                ant_task = root_node
                time = self.start_time

                adt.visit_node(ant, ant_task, time)
                step = 1

                # Generate Path for each ant
                while not ant._search_complete:

                    # Determine which nodes the ant can visit
                    valid_choices = adt.ant_task_choices(ant, interval)

                    # Check for Path Termination (e.g. empty list or exceeded horizon)
                    if not valid_choices:
                        ant._search_complete = True
                        if self._local_pheromone_update():
                            adt.update_pheromones([ant], self._objective)

                        break

                    # Check if the ant has taken too much time
                    elif ant.simulated_time >= self.end_time:
                        if self._local_pheromone_update():
                            adt.update_pheromones([ant], self._fitness)
                        break

                    # Make move
                    time += ant_task.wcet
                    ant_task = self._edge_selection(ant, valid_choices, adt, i+1)

                    adt.visit_node(ant, ant_task, time)

                    step += 1

                # Termination by Duration - Stop the rest of the swarm from searching
                if self._flag_termination_by_duration and self._termination_by_duration(start_runtime):
                    break

            colony.sort(key=lambda ant: self._fitness(ant.get_schedule()), reverse=True)

            if self._flag_local_pheromone_update == self.ANT_COLONY:
                pass

            global_solutions = global_solutions + colony
            global_solutions.sort(key=lambda ant: self._fitness(ant.get_schedule()), reverse=True)
            global_solutions = global_solutions[:len(colony)]
            new_best_schedule_value = self._fitness(global_solutions[0].get_schedule())

            if self.verbose:
                print("Iterative Best: " + str(self._fitness(colony[0].get_schedule())) + " | Best so far: " +
                      str(self._fitness(global_solutions[0].get_schedule())))

            # Termination by Duration
            if self._flag_termination_by_duration and self._termination_by_duration(start_runtime):
                break

            # Termination by Max Iteration
            if self._flag_termination_by_max_iteration and self._termination_by_max_iterations(i):
                print("Max iterations met" + str(i) + " | " + str(self.max_iterations))
                break

            # Termination by Generational Delta
            if self._flag_termination_by_generational_delta and \
                    self._termination_by_generational_delta(current_best_schedule_value, new_best_schedule_value):
                print("Generational Delta Threshold Met")
                print("Convergence Met after " + str(i) + " iterations")
                break

            # Prep Next iteration
            i += 1
            current_best_schedule_value = global_solutions[0]

            # Update Pheromone Matrix

            # Update using Full Colony
            if self._full_colony_update():
                adt.update_pheromones(colony, self._fitness)

            elif self._elitism():
                solutions = colony + [global_solutions[0]]
                adt.update_pheromones(solutions, self._fitness)

            if self._best_only_update():
                update_size = 1

                # Update Using Iterative-Best
                if self._iterative_best_update():
                    if self._ranked():
                        update_size = 1
                        adt.update_pheromones(colony[0:update_size], self._fitness())
                    else:
                        adt.update_pheromones([colony[0]], self._fitness())

                # Update using Best-So-Far
                if self._global_best_update():
                    if self._ranked():
                        update_size = 1
                        adt.update_pheromones(global_solutions[0:update_size], self._fitness())
                    adt.update_pheromones([global_solutions[0]], self._fitness())



        print("Finished after " + str(i) + " swarms")

        best_solution = global_solutions[0].get_schedule() # Remove initial root_node

        return best_solution # Skip the root_node of the best_schedule

    def _edge_selection(self, ant, choices, adt, iteration=1):
        """ Choices then next node to visit (and hence the edge to use).

        :param ant:
        :param choices: List of valid choices
        :return:
        """

        if random.uniform(0, 1) < self.epsilon / iteration:
            return choices[random.randint(0, len(choices) - 1)]
        else:
            probabilities = []
            last = ant.last_visited_node()
            alpha = self.alpha + iteration/10
            beta = self.beta / iteration
            for choice in choices:
                e = (last, choice)

                # Get Pheromone Value
                p_value = adt.get_pheromone_value(e)
                p_value = p_value ** alpha

                # Get Heuristic Value
                h_value = self._attractiveness(choice, last[1])

                # Higher probability of negative values (bad solution)
                if h_value < 0:    # Stop creation of Complex Numbers
                    h_value = -1 * h_value
                    h_value = -1 * (h_value ** beta)
                elif h_value != 0:  # Stop Divide by Zero Error
                    h_value = h_value ** beta

                value = p_value * h_value

                probabilities.append(value)

            # Shift probabilities to remove negative values
            smallest = min(probabilities)
            if smallest < 0:
                probabilities = [p - smallest for p in probabilities]

            # Normalize
            norm = sum(probabilities)

            if norm == 0: # Fringe case when all values are awful
                return random.choice(choices)
            else:
                probabilities = [p/norm for p in probabilities]

                # Make a weighted Selection
                node_choice = random.choices(choices, weights=probabilities)
                return node_choice[0]

    def _attractiveness(self, Node: AntTask, time: datetime):
        """ Heuristic function that takes into consideration:
        (1) the amount of value lost if the task is taken early

        :param Node:
        :param time:
        :return:
        """
        if Node.is_sleep_task():
            return 0

        value_now = Node.value(timestamp=time)
        utopia_point = Node.soft_deadline
       #hard_deadline = Node.hard_deadline.timestamp()

        if time < utopia_point:
            utopia_value = Node.value(timestamp=utopia_point)
            return value_now - (utopia_value - value_now)
        else:
            return value_now

    def _early_penalty(self, node: AntTask, current_value: float):
        return node.soft_deadline.timestamp() - current_value

    def _fitness(self, schedule):
        """ Provides a value for the given schedule. For invalid schedules, the "invalid schedule value" is returned.
        Otherwise, the simulated value is returned.

        :param schedule:
        :return:
        """
        if not self._validate_schedule(schedule, self.optimization_horizon):
            return self.invalid_schedule_value
        else:
            return self.simulate_execution(schedule)

    def algorithm_name(self):
        return "Ant System"

    # TEMPLATE METHODS #

    def _full_colony_update(self):
        return True
    def _local_pheromone_update(self):
        return False

    def _iterative_best_update(self):
        return False

    def _global_best_update(self):
        return False

    def _elitism(self):
        return False

    def _best_only_update(self):
        return False

    def _ranked(self):
        return False

    def _min_max_bounds(self):
        return False


class ElitistAntScheduler(AntScheduler):

    def _full_colony_update(self):
        return False

    def _elitism(self):
        return True

    def algorithm_name(self):
        return "Elitist Ant"


class AntColonyScheduler(AntScheduler):

    def _local_pheromone_update(self):
        return True

    def _full_colony_update(self):
        return False

    def _best_only_update(self):
        return True

    def algorithm_name(self):
        return "Ant Colony Optimization (ACO)"


class Ant:

    def __init__(self):
        """

        """
        self._search_complete = False
        self._path = []                 # [(AntTask, timestamp)]
        self._ant_tasks = []            # [(AntTask)]
        self._schedule: List[AbstractTask] = []             # [AbstractTasks]
        self._path_value = None
        self._simulated_time = None

    @property
    def simulated_time(self):
        return self._simulated_time

    def get_visited_nodes(self):
        """ Returns an iterator of all the nodes (AntTasks, timestamp) the ant has visited.
            Note that a "node" in this context consists of a AntTasks and a Timestamp

        :return: Iterator of (AntTask, timestamp)
        """
        return map(lambda x: x, self._path)

    def get_completed_ant_tasks(self):
        """ Returns an iterator of all the AntTasks the ant has completed.

        :return: Iterator
        """
        return map(lambda x: x[0], self._path)

    def get_schedule(self):
        """ Unpacks the AntTasks and returns a list of the original tasks

        :return:
        """
        return Schedule([x[0]._task for x in self._path])  #list(map(lambda x: x[0]._task, self._path))

    def last_visited_node(self):
        visited_nodes = list(self.get_visited_nodes())
        if not visited_nodes: # empty case
            return None

        return visited_nodes[-1]

    def visit(self, ant_task: AntTask, timestamp: datetime):
        """

        :param ant_task:
        :param timestamp:
        :return:
        """
        self._simulated_time = timestamp
        node = (ant_task, timestamp)
        self._path.append(node)
        self._ant_tasks.append(ant_task)
        self._schedule.append(ant_task._task)

    def get_path_value(self):
        """

        :return:
        """
        if not self._search_complete:
            raise ValueError("Ant has not completed path")

        if self._path_value is None:
            self._path_value = 0


class AntDependencyTree:

    def __init__(self, tasklist: List[AbstractTask], **kwargs):
        """

        :param tasklist:
        """
        self._ant_tasks = []
        # self._pheromones: Dict[List[float]] = {}
        self._pheromomnes: Dict[float] = {}
        self._edge_visits: Dict[Tuple[int,int]] = {}
        self.initial_pheromone_amount = 2
        self.min_pheromone_amount = kwargs.get("min_pheromone_value", 1)
        self.max_pheromone_amount = kwargs.get("max_pheromone_value", 100)
        self._evaporation_rate = kwargs.get("evaporation_rate", 0.2)
        self._flag_max_min_ant_system = kwargs.get("min_max", False)

        for task in tasklist:
            self._ant_tasks.append(AntTask(task))

        self._update_dependencies()

    def get_ant_task(self, task: AbstractTask):
        """

        :param task:
        :return:
        """
        node = list(filter(lambda x: x._task is task, self._ant_tasks))
        return node[0]

    def _update_dependencies(self):
        """Fixes the dependency references so that to point to the Ant Task Wrapper
        instead of the owned (composite) task class

        :return: None
        """

        for node in self._ant_tasks:
            dependencies = node._task.get_dependencies()

            for dependency in dependencies:
                found = False
                for other_node in self._ant_tasks:
                    if dependency is other_node._task:
                        found = True
                        node.add_dependency(other_node)
                        break

                if not found:
                    raise ValueError("Dependency Error")

    def ant_task_choices(self, ant: Ant, sleep_interval: timedelta):
        """ Determines which nodes are available to the ant. In other words, it only excludes nodes (tasks) whose
            dependencies haven't been met or if it has already been visited by the ant.

        :param ant: Ant
        :param interval: Only Used to dynamically create SleepTasks as needed
        :return: List(AntTasks)
        """
        valid_choices = []
        completed_ant_tasks = list(ant.get_completed_ant_tasks())

        # Remove ant_tasks that have already been visited
        choices = [ant_task for ant_task in self._ant_tasks if ant_task not in completed_ant_tasks]

        # Only add ant_tasks if their dependent nodes have already been visited
        for choice in choices:
            if all(map(lambda x: ant in x.visited_by,  choice.get_dependencies())):
                valid_choices.append(choice)

        # If there are still nodes to visit, add SleepTask
        if not not valid_choices:
            st = SleepTask(wcet=sleep_interval)#runtime=sleep_interval, analysis_type="SLEEPANALYSIS", wcet=sleep_interval)
            valid_choices.append(AntTask(st))

        return valid_choices

    def visit_node(self, ant: Ant, ant_task: AntTask, timestamp=None):
        """
        :param ant:
        :param ant_task:
        :param timestamp:
        :return:
        """

        previous_node = ant.last_visited_node()

        ant_task.accept(ant, timestamp)



        # Update Pheromone Matrix
        # edge = (ant.last_visited_node(), (node, timestamp))

        #self._pheromones[edge] = self._pheromones.get(edge, 0) + 10000

    def update_pheromones(self, ants: List[Ant], objective):
        """

        :param ant: the Ant
        :param fitness: A Fitness Function
        :return:
        """
        new_pheromones: Dict[float] = {}

        # Calculate new pheromones to add
        for ant in ants:
            schedule = ant.get_schedule()

            value = objective(schedule)

            path = list(ant.get_visited_nodes())
            for i, node in enumerate(path[:-1]):
                key = (path[i], path[i+1])

                current = new_pheromones.get(key, self.initial_pheromone_amount)
                new_pheromones[key] = current + value

        for key, value in new_pheromones.items():
            current = self._pheromomnes.get(key, 0)
            self._pheromomnes[key] = (1-self._evaporation_rate) * current + self._evaporation_rate * value

            # Limits pheromone values between [min, max]
            if self._flag_max_min_ant_system:
                if self._pheromomnes[key] < self.min_pheromone_amount:
                    self._pheromomnes[key] = self.min_pheromone_amount
                if self._pheromomnes[key] > self.max_pheromone_amount:
                    self.min_pheromone_amount = self.max_pheromone_amount

        """
            # Empty Case
            if key not in self._pheromones:
                self._pheromones[key] = []

            if key not in self._edge_visits:
                self._edge_visits[key] = 0

            self._pheromones[key].append(value)
            self._edge_visits[key] = self._edge_visits[key] + 1
        """

    def get_pheromone_value(self, key):
        """

        :param key:
        :return:
        """

        """
                if key not in self._pheromones:
                    return self.min_pheromone_amount
                else:
                    return sum(self._pheromones[key]) / self._edge_visits[key]
                """

        return self._pheromomnes.get(key, self.initial_pheromone_amount)

    def get_saturated_path(self, root_task, start_time, end_time, interval):
        """

        :return:
        """
        path = []
        greedy_ant = Ant()
        ant_task = root_task
        time = start_time.timestamp()
        self.visit_node(greedy_ant, ant_task, time)
        while not greedy_ant._search_complete:

            # Determine which nodes the ant can visit
            valid_choices = self.ant_task_choices(greedy_ant, interval)

            # Check for Path Termination (e.g. empty list or exceeded horizon)
            if not valid_choices:
                greedy_ant._search_complete = True
                break

            # Check if the ant has taken too much time
            elif greedy_ant.simulated_time >= end_time.timestamp():
                break

            # Make Greedy move
            next_ant_task = self._greedy_edge_selection(greedy_ant, valid_choices, self, interval + 1)
            time += ant_task.wcet
            self.visit_node(greedy_ant, next_ant_task, time)

        return greedy_ant.get_schedule()

    def _greedy_edge_selection(self, ant, choices, adt, iteration=1):
        """ Choices then next node to visit (and hence the edge to use).

        :param ant:
        :param choices: List of valid choices
        :return:
        """

        last = ant.last_visited_node()
        time = last[1] + last[0].wcet
        edges = []
        for choice in choices:
            e = (last, (choice, time))

            # Get Pheromone Value
            p_value = adt.get_pheromone_value(e)

            edges.append((e, p_value))

        return max(edges, key=itemgetter(1))[0][1][0]

    '''
    def evaporate_pheromones(self, evaporation_rate=0.8):
        """

        :param evaporation_rate:
        :return:
        """
        for k,v in self._pheromones.items():
            self._pheromones[k] = list(map(lambda x: x * evaporation_rate, v))
    '''


class SimulateAnnealingScheduler(MetaHeuristicScheduler):

    linear_decay = 0
    geometric_decay = 1
    slow_decay = 2

    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super().__init__(**kwargs)
        self.temperature = kwargs.get("temperature", 50)
        self.solution_space_size = kwargs.get("solution_space_size", 10000)
        self.decay_method = kwargs.get("decay_method", SimulateAnnealingScheduler.geometric_decay)
        self.decay_constant = kwargs.get("decay_constant", 0.9)
        self.elitism = True
        self.local_searches = 10
        self.best_solution = None
        self.best_solution_value = 0

    def schedule_tasks(self, tasklist: List[AbstractTask], interval: int) -> List[AbstractTask]:
        np.seterr(all='raise')
        print("Generating Schedule with Simulated Annealing")
        start_runtime = datetime.now()

        self._converged = False
        #new_task_list = self._initialize_tasklist(tasklist, interval)

        # Set Starting State
        current_state = self.generate_random_schedule(tasklist, interval)#random.sample(new_task_list, len(new_task_list))

        # Set Best Solution
        self.best_solution = current_state
        self.best_solution_value = self.simulate_execution(self.best_solution)

        i = 0
        temp = self._initialize_temperature()
        while not self._converged:

            # Perform searches at current temp
            current_state_energy = self._energy(current_state)

            for i in range(self.local_searches):
                neighbor = self._neighbor(current_state)
                neighbor_energy = self._energy(neighbor)

                # Possibly change state
                if self._acceptance_probability(current_state_energy, neighbor_energy, temp) >= random.uniform(0, 1):
                    current_state = neighbor
                    current_state_energy = neighbor_energy

                    if self.elitism:
                        old_best_value = self.best_solution_value
                        current_state_value = self.simulate_execution(current_state)

                        # Update Best Solution
                        if self.best_solution_value < current_state_value:
                            self.best_solution = current_state
                            self.best_solution_value = current_state_value

                        # Termination by Generational Delta
                        if self._flag_termination_by_generational_delta and \
                                self._termination_by_generational_delta(old_best_value,
                                                                        self.best_solution_value):  # Checks delta
                            print("Generational Delta Threshold Met")
                            print("Convergence Met after " + str(i) + " iterations")
                            self._converged = True
                            break

                # Termination by Duration
                if self._flag_termination_by_duration and self._termination_by_duration(start_runtime):
                    break


            # Termination by Max Iteration
            if self._flag_termination_by_max_iteration and self._termination_by_max_iterations(i):
                print("Max iterations met" + str(i) + " | " + str(self.max_iterations))
                break

            i += 1

            # Termination by Duration
            if self._flag_termination_by_duration and self._termination_by_duration(start_runtime):
                 break

            # Update temperature
            temp = self._update_temperature(temp, self.decay_method)

        if self.elitism:
            return self.best_solution
        else:
            return current_state

    def _update_temperature(self, temp, decay_method=slow_decay):
        """

        :param temp:
        :param decay_method:
        :return:
        """
        if decay_method == SimulateAnnealingScheduler.geometric_decay:
            return temp * self.decay_constant
        elif decay_method == SimulateAnnealingScheduler.linear_decay:
            return temp - self.decay_constant
        elif decay_method == SimulateAnnealingScheduler.slow_decay:
            return temp/(1 + self.decay_constant * temp)
        else:
            raise ValueError("Invalid Decay Method Used")

    def _neighbor(self, state):
        """

        :param state:
        :param solution_space:
        :return:
        """
        candidate = state[:]

        # Choose a permutation method at random
        method = random.choice([
            self.permute_schedule_by_blockinsert,
            self.permute_schedule_by_inverse,
            self.permute_schdule_by_swap,
            self.permute_schedule_by_transport
        ])
        method(candidate)

        return candidate

    def _acceptance_probability(self, energy: float, energy_new: float, temp):
        """ Determines the probability of accepting the new state based on its energy.

        :param energy: energy value of the current state
        :param energy_new: energey value of the new state
        :param temp: Current Temperature
        :return: Returns the probability of accepting the new state.
        """

        if energy_new <= energy:
            return 1
        else:
            change = energy_new - energy
            x = sympy.Rational(-change,temp)
            p: float = exp(x)
            return p

    def _energy(self, state):
        val = self._objective(state)

        # Check for zero val case
        if val == 0:
            return -(self.invalid_schedule_value)
        else:
            return sympy.Rational(1, val)

    def _initialize_temperature(self, **kwargs):
        return self.temperature


class EnhancedListBasedSimulatedAnnealingScheduler(SimulateAnnealingScheduler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temperature_list_length = 25
        self.local_searches = 20
        self.pos = 0.4
        self._temperatures = TemperatureQueue()
        self.num_agents = 20

    def schedule_tasks(self, tasklist: Schedule, sleep_interval: timedelta) -> List[AbstractTask]:
        print("Generating Schedule Using " + self.algorithm_name() + " Algorithm")
        start_runtime = datetime.now()

        # Set Starting State
        current_state = self.generate_random_schedule(tasklist, sleep_interval) #random.sample(new_task_list, len(new_task_list))
        current_state_energy = self._energy(current_state)

        # Initialize agent lists
        agent_solution: List[Schedule] = []
        agent_task_index: List[int] = []

        # Initialize the temperature list
        # Start with a list that is twice as large
        self._temperatures.reset()
        cut = floor(self.temperature_list_length / 2)
        for i in range(self.temperature_list_length * 2):

            # Add temperature
            neighbor_state = self._neighbor(current_state, tasklist, sleep_interval)
            neighbor_state_energy = self._energy(neighbor_state)
            temp = abs(current_state_energy - neighbor_state_energy)
            self._temperatures.push(temp)

            # Change states if the neighbor is better
            if neighbor_state_energy < current_state_energy:
                current_state = neighbor_state
                current_state_energy = self._energy(current_state)

        # Remove top and bottom of the temperature list to restore original size
        # e.g. remove noise
        self._temperatures.remove_bottom(cut)
        self._temperatures.remove_top(cut)

        # Initialize agents
        # Note, number of agents is equal to the number of populations
        for agent_id in range(self.num_agents):
            neighbor_state = self._neighbor(current_state, tasklist, sleep_interval)
            agent_solution.append(neighbor_state)
            agent_task_index.append(random.randint(0, len(neighbor_state)))

        if self._temperatures.peek() == 0:
            self._temperatures.push(100)  # temp solution
            #raise RuntimeError("Unable to initialize temperatures. Please consider increasing the temperature list length.")

        # Set Best Solution
        self.best_solution = max(agent_solution, key=lambda x:  self._objective(x))

        while True:

            # Loop through agents
            for agent_id in range(self.num_agents):
                current_state = agent_solution[agent_id]
                temp = self._temperatures.peek()

                num_accepted_worse_solutions = 0
                total_temperatures = 0

                # Calculate agent_mcl on each pass
                agent_mcl: int = self._agent_mcl(self.pos, self.local_searches, len(tasklist), start_runtime, self.learning_duration)

                for i in range(agent_mcl):

                    # update agent task index
                    agent_task_index[agent_id] = (agent_task_index[agent_id] +1) % len(current_state)
                    index = agent_task_index[agent_id]
                    # Generate Candidate Solution
                    neighbor_state = self._neighbor(current_state, tasklist, sleep_interval, agent_task_index[agent_id])

                    # Calculate probability of accepting candidate
                    current_state_energy = self._energy(current_state)
                    neighbor_state_energy = self._energy(neighbor_state)
                    prob = self._acceptance_probability(current_state_energy, neighbor_state_energy, temp)

                    # Check if candidate is accepted
                    rand = random.uniform(0,1)
                    if prob > rand:

                        # Update bad solution acceptance variables
                        change = neighbor_state_energy - current_state_energy
                        if change > 0:
                            total_temperatures = total_temperatures + -change/log(rand)
                            num_accepted_worse_solutions = num_accepted_worse_solutions + 1

                        # Update Best Solution
                        elif current_state_energy < self._energy(self.best_solution):
                            self.best_solution = current_state

                        current_state = neighbor_state

                    # Update Temperature list
                    if num_accepted_worse_solutions > 0:
                        new_temp = total_temperatures / num_accepted_worse_solutions
                        self._temperatures.pop()                # Remove old max_temp
                        self._temperatures.push(new_temp)       # Add new temp

                    # Termination by Duration
                    # Break out of local searches
                    if self._flag_termination_by_duration and self._termination_by_duration(start_runtime):
                        return self.best_solution

        return self.fit_to_horizon(self.best_solution)

    def _neighbor(self, state, tasklist, sleep_interval, fixed_task_index = None):
        """

        :param state:
        :param solution_space:
        :return:
        """
        candidate1 = Schedule(state)
        candidate2 = Schedule(state)
        candidate3 = Schedule(state)
        candidate4 = Schedule(state)
        candidate5 = self.generate_random_schedule(tasklist, sleep_interval)

        if fixed_task_index is None:
            self.permute_schedule_by_blockinsert(candidate1)
            self.permute_schedule_by_inverse(candidate2)
            self.permute_schdule_by_swap(candidate3)
            self.permute_schedule_by_transport(candidate4)
        else:
            self.permute_schedule_by_blockinsert(candidate1, fixed_task_index)
            self.permute_schedule_by_inverse(candidate2, fixed_task_index)
            self.permute_schdule_by_swap(candidate3, fixed_task_index)
            self.permute_schedule_by_transport(candidate4, fixed_task_index)


        candidates = [candidate1, candidate2, candidate3, candidate4, candidate5]

        best_candidate = max(candidates, key=lambda x: self._objective(x))

        return best_candidate

    def _agent_mcl(self, pos, mcl, tasklist_size: int, start_runtime, learning_duration) -> int:
        """

        :param pos:
        :param mcl:
        :param tasklist_size:
        :param start_runtime:
        :param duration:
        :return:
        """
        elapsed_time: timedelta = datetime.now() - start_runtime
        #norm_elapsed_time: float = elapsed_time.total_seconds() / (duration * 60)
        #elapsed_time = elapsed_time.total_seconds()

        MAX_START: timedelta = timedelta(seconds=pos * learning_duration.total_seconds())
        MAX_END: timedelta = timedelta(seconds=(1 - pos) * learning_duration.total_seconds())

        # Early Case
        if elapsed_time < MAX_START:
            weight: float = max( (elapsed_time.total_seconds() / MAX_START.total_seconds()) * 3, 1)
            vmlc = ceil(weight * (mcl / 2))
            if self.verbose:
                print("early: " + str(vmlc))
            return vmlc
        elif MAX_START < elapsed_time < MAX_END:
            vmlc = ceil(3 * mcl/2)
            if self.verbose:
                print("max: " + str(vmlc))
            return vmlc
        else:   # Case elapsed_time > MAX_END
            end_time = learning_duration
            period = end_time - MAX_END
            weight: float = max((period.total_seconds() - (elapsed_time.total_seconds() - MAX_END.total_seconds())) / period.total_seconds() * 3, 1)
            vmlc = ceil(weight * (mcl / 2))
            if self.verbose:
                print("late: " + str(vmlc))
            return vmlc

    def algorithm_name(self):
        return "Enhanced List-Based Simulated Annealing"


class TemperatureQueue():
    def __init__(self):
        self._temperatures = []
        heapq.heapify(self._temperatures)

    def __len__(self):
        return len(self._temperatures)

    def reset(self):
        self._temperatures = []
        heapq.heapify(self._temperatures)

    def pop(self):
        temp = heapq.heappop(self._temperatures)
        return -temp

    def push(self, temp):
        heapq.heappush(self._temperatures, -temp)

    def remove_top(self, n: int):
        """ Removes the n largest temperatures

        :param n: The number of temperatures to remove
        """
        self._temperatures.sort()
        del self._temperatures[:n]
        heapq.heapify(self._temperatures)

    def remove_bottom(self, n: int):
        self._temperatures.sort()
        del self._temperatures[-n:]
        heapq.heapify(self._temperatures)

    def peek(self):
        x = heapq.nsmallest(1, self._temperatures)
        return -x[0]
