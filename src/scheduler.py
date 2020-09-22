from abc import ABC, abstractmethod
from task import AbstractTask, DummyTask#, ScheduledTask
from datetime import datetime
from typing import List
from math import ceil
import random


class SchedulerFactory:

    @staticmethod
    def get_scheduler(scheduler_type: str):
        scheduler = None

        if scheduler_type.upper() == "GENETIC":
            scheduler = GeneticScheduler()
        elif scheduler_type.upper() == "DUMMY":
            scheduler = DummyScheduler()
        else:
            raise Exception("Invalid Analysis Type")

        return scheduler


class AbstractScheduler(ABC):

    def __init__(self):
        pass

    def generate_dummy_tasks(self, tasklist: List[AbstractTask], interval):
        # Calculate the number of dummy tasks needed
        last_interval = max(task.hard_deadline.timestamp() + task.wcet for task in tasklist)
        total_wcet = sum(task.wcet for task in tasklist)
        dif = last_interval + total_wcet - datetime.now().timestamp()
        num_dummy_tasks = ceil(dif / interval)

        # Generated Scheduled Dummy Tasks
        dummy_tasks = []
        for x in range(num_dummy_tasks):
            dummy_tasks.append(DummyTask(None, runtime=interval))

        return tasklist + dummy_tasks

    '''Checks if Schedule is consistent with dependencies (e.g. no task is scheduled before any of its dependencies)
    
    '''
    def _validate_schedule(self, tasklist: List[AbstractTask]) -> bool:

        result = True
        non_dummy_tasks = [task for task in tasklist if not task.is_dummy()]
        prior_tasks = []
        # Check Each scheduledTask
        for i, task in enumerate(non_dummy_tasks):

            # Check Dependencies
            if task.has_dependencies():
                dependencies = task.get_dependencies()
                if not self._verify_dependencies(dependencies, prior_tasks):
                    return False

            prior_tasks.append(task)

        return True

    def _verify_dependencies(self, tasklist: List[AbstractTask], prior_tasks) -> bool:

        for dependency in tasklist:
            if not self._verify_dependency(dependency, prior_tasks):
                return False

        return True

    ''' Verifies each dependency in dependency list is in tasklist'''

    def _verify_dependency(self, dependency: AbstractTask, prior_tasks: List[AbstractTask]) -> bool:

        if dependency in prior_tasks:
            return True
        else:
            return False

    @abstractmethod
    def schedule_tasks(self, tasklist: List[AbstractTask], interval: int) -> List[AbstractTask]:
        pass

    def _simulate_execution(self, tasklist: List[AbstractTask]):
        time = datetime.now().timestamp()
        total_value = 0

        for task in tasklist:
            if not task.is_dummy():
                total_value += task.value(timestamp=time)

            time += task.wcet

        return total_value


class DummyScheduler(AbstractScheduler):

    def __init__(self):
        super().__init__()

    def schedule_tasks(self, tasklist: List[AbstractTask], interval) -> List[AbstractTask]:
        new_tasklist = self.generate_dummy_tasks(tasklist, interval)
        max_iteration = 100
        valid = False
        schedule = None
        i = 0

        while not valid:
            schedule = random.sample(new_tasklist, len(new_tasklist))
            valid = self._validate_schedule(schedule)

            list = [(i,task, task.dependent_tasks) for i, task in enumerate(schedule) if not task.is_dummy()]
            if valid:
                print("===== Valid Schedule =====")
                break
            elif i >= max_iteration:
                raise ValueError("Failed to generate a valid schedule")

            i += 1

        return schedule


class GeneticScheduler(AbstractScheduler):

    def __init__(self, **kwargs):
        super().__init__()
        if "max_generations" in kwargs:
            self._max_generations = kwargs.get("max_generations")
        else:
            self._max_generations = 1000

        if "population_size" in kwargs:
            self._population_size = kwargs.get("population_size")
        else:
            self._population_size = 1000

        self._invalid_schedule_value = -1000.0
        self._breeding_percentage = .05

    def schedule_tasks(self, tasklist: List[AbstractTask], interval) -> List[AbstractTask]:

        population = []

        # Initialize Population
        for x in range(self._population_size):
            population.append(random.sample(tasklist, len(tasklist)))

        for x in range(self._max_generations):
            values = []
            for schedule in population:
                values.append(self.fitness(schedule))
            sample = self.selection(population, values)
            self.crossover()
            self.mutation()

        return tasklist

    def fitness(self, schedule):

        if not self._validate_schedule(schedule):
            return self._invalid_schedule_value
        else:
            return self._simulate_execution(schedule)

    def selection(self, population, value):
        sample = list(zip(population, value))
        sorted_sample = sorted(sample,
                               key=lambda item: item[1],
                               reverse=True)
        new_sorted_sample, _ = list(zip(*sorted_sample))
        cutoff = ceil(len(sorted_sample)*self._breeding_percentage)
        parent_sample = new_sorted_sample[:cutoff]

        return parent_sample

    def crossover(self):
        pass

    def mutation(self):
        pass
