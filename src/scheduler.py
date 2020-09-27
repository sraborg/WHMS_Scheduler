from abc import ABC, abstractmethod
from task import AbstractTask, DummyTask #, ScheduledTask
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
        self._optimization_horizon = None
        pass

    def generate_sleep_tasks(self, tasklist: List[AbstractTask], interval):

        # Calculate the number of dummy tasks needed
        last_interval = max(task.hard_deadline.timestamp() + task.wcet for task in tasklist)
        total_wcet = sum(task.wcet for task in tasklist)
        dif = last_interval + total_wcet - datetime.now().timestamp()
        num_dummy_tasks = ceil(dif / interval)

        # Generated Scheduled Dummy Tasks
        sleep_tasks = []
        for x in range(num_dummy_tasks):
            sleep_tasks.append(DummyTask(None, analysis_type="SleepAnalysis", wcet=interval))

        return tasklist + sleep_tasks

    '''Checks if Schedule is consistent with dependencies (e.g. no task is scheduled before any of its dependencies)
    
    '''
    def _validate_schedule(self, tasklist: List[AbstractTask]) -> bool:

        # Check for Duplicates
        if self._no_duplicate_tasks(tasklist):
            return False

        # Check Dependencies
        if not self._verify_schedule_dependencies(tasklist):
            return False

        return True

    '''Checks that every dependency for every task is scheduled prior to the task

    '''
    def _verify_schedule_dependencies(self, schedule: List[AbstractTask]):
        non_sleep_tasks = [task for task in schedule if not task.is_sleep_task()]
        prior_tasks = []

        # Check Each scheduledTask
        for i, task in enumerate(non_sleep_tasks):

            # Check Dependencies
            if not self._verify_task_dependencies(task, prior_tasks):
                return False

            prior_tasks.append(task)

        return True

    '''Checks that every dependency for a task is scheduled prior to the task.

        '''
    def _verify_task_dependencies(self, task, prior_tasks):

        if task.has_dependencies():
            dependencies = task.get_dependencies()

            for dependency in dependencies:
                if dependency not in prior_tasks:
                    return False
        return True

    '''Checks that every dependency for a task is scheduled prior to the task.
    
    def _verify_dependencies(self, tasklist: List[AbstractTask], prior_tasks) -> bool:

        for dependency in tasklist:
            if not self._verify_dependency(dependency, prior_tasks):
                return False

        return True

    def _verify_dependency(self, dependency: AbstractTask, prior_tasks: List[AbstractTask]) -> bool:

        if dependency in prior_tasks:
            return True
        else:
            return False
   
    '''

    def _no_duplicate_tasks(self, tasklist: List[AbstractTask]):
        for task in tasklist:

            # ingore sleepTasks
            #if task.is_dummy():
            #   continue

            if tasklist.count(task) > 1:
                return False

        return True

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
        new_tasklist = self.generate_sleep_tasks(tasklist, interval)
        max_iteration = 100
        valid = False
        schedule = None
        i = 0

        print("Generating Schedule at Random")
        while not valid:
            schedule = random.sample(new_tasklist, len(new_tasklist))
            valid = self._validate_schedule(schedule)

            list = [(i,task, task.dependent_tasks) for i, task in enumerate(schedule) if not task.is_dummy()]
            if valid:
                print("===== Valid Schedule =====")
                break
            elif i >= max_iteration:
                raise ValueError("Failed to generate a valid schedule after " + str(max_iteration) + " attempts")

            i += 1

        return schedule


class GeneticScheduler(AbstractScheduler):

    def __init__(self, **kwargs):
        super().__init__()
        if "max_generations" in kwargs:
            self._max_generations = kwargs.get("max_generations")
        else:
            self._max_generations = 50

        if "population_size" in kwargs:
            self._population_size = kwargs.get("population_size")
        else:
            self._population_size = 10000

        self._invalid_schedule_value = -1000.0
        self._breeding_percentage = .05
        self._mutation_rate = 0.01
        self._tasks = None

    def schedule_tasks(self, tasklist: List[AbstractTask], interval) -> List[AbstractTask]:

        self._tasks = tasklist
        new_task_list = self.generate_sleep_tasks(tasklist, interval)
        population = []
        print("Generating Schedule Using Genetic Algorithm")

        # Initialize Population
        for x in range(self._population_size):
            population.append(random.sample(new_task_list, len(new_task_list)))

        for x in range(self._max_generations):
            print("Processing Generation " + str(x+1))

            breeding_sample = self.selection(population)
            next_generation = self.crossover(breeding_sample, len(population))
            population = self.mutation(next_generation)

        best_fit = self.selection(population)[0]
        if not self._validate_schedule(best_fit):
            raise ValueError("Failed Generate Schedule after " + str(self._max_generations) + " generations")

        return best_fit

    def fitness(self, schedule):

        if not self._validate_schedule(schedule):
            return self._invalid_schedule_value
        elif self._all_tasks_present(schedule, self._tasks):
            return self._invalid_schedule_value
        else:
            return self._simulate_execution(schedule)

    def selection(self, population, **kwargs):
        values = []
        for schedule in population:
            values.append(self.fitness(schedule))

        sample = list(zip(population, values))
        sorted_sample = sorted(sample,
                               key=lambda item: item[1],
                               reverse=True)
        new_sorted_sample, _ = list(zip(*sorted_sample))
        cutoff = ceil(len(sorted_sample)*self._breeding_percentage)
        parent_sample = new_sorted_sample[:cutoff]

        return parent_sample

    def crossover(self, parents, population_size):
        next_generation = []

        for x in range(population_size):
            p1, p2 = random.sample(parents, 2)
            midpoint = ceil(len(p1)/2)
            child = p1[:midpoint] + p2[midpoint:]
            next_generation.append(child)

        return next_generation

    def mutation(self, population):

        for schedule in population:
            for i, task in enumerate(schedule):
                if random.random() <= self._mutation_rate:
                    next_index = random.randint(0, len(schedule)-1)
                    temp = task
                    schedule[i] = schedule[next_index]
                    schedule[next_index] = temp

        return population

    def _all_tasks_present(self, schedule, tasklist):
        for task in tasklist:
            if task not in schedule:
                return False

        return True