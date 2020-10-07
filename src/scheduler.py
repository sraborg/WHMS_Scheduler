from abc import ABC, abstractmethod
import numpy as np
from task import AbstractTask, DummyTask, SleepTask, AntTask #, ScheduledTask
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
        elif scheduler_type.upper() == "RANDOM":
            scheduler = RandomScheduler()
        else:
            raise Exception("Invalid Analysis Type")

        return scheduler


class AbstractScheduler(ABC):

    def __init__(self):
        self._optimization_horizon = None
        self.start_time = None
        self.verbose = True

    @property
    def optimization_horizon(self):
        if self._optimization_horizon is None:
            return

    @staticmethod
    def calculate_optimization_horizon(tasklist):
        horizon = max(task.hard_deadline.timestamp() + task.wcet for task in tasklist)
        return horizon

    def _initialize_tasklist(self, tasklist: List[AbstractTask], interval):
        self._tasks = tasklist
        return tasklist + self.generate_sleep_tasks(tasklist, interval)

    def generate_sleep_tasks(self, tasklist: List[AbstractTask], interval, **kwargs):

        optimization_horizon = None
        calculated_horizon = AbstractScheduler.calculate_optimization_horizon(tasklist)

        if self._optimization_horizon is None or calculated_horizon > self._optimization_horizon:
            optimization_horizon = calculated_horizon
        else:
            optimization_horizon = self._optimization_horizon

        if "start_time" in kwargs:
            start_time = kwargs.get("start_time")
        else:
            start_time = datetime.now()

        # Calculate the number of dummy tasks needed
        total_wcet = sum(task.wcet for task in tasklist)
        dif = (optimization_horizon - start_time.timestamp()) - total_wcet
        num_sleep_tasks = ceil(dif / interval)

        # Generated Scheduled Dummy Tasks
        sleep_tasks = []
        for x in range(num_sleep_tasks):
            sleep_tasks.append(SleepTask(None, wcet=interval))

        return sleep_tasks

    '''Checks if Schedule is consistent with dependencies (e.g. no task is scheduled before any of its dependencies)
    
    '''
    @staticmethod
    def _validate_schedule(tasklist: List[AbstractTask]) -> bool:

        # Check for Duplicates
        if not AbstractScheduler._no_duplicate_tasks(tasklist):
            return False

        # Check Dependencies
        if not AbstractScheduler._verify_schedule_dependencies(tasklist):
            return False

        return True

    '''Checks that every dependency for every task is scheduled prior to the task

    '''
    @staticmethod
    def _verify_schedule_dependencies(schedule: List[AbstractTask]):
        non_sleep_tasks = [task for task in schedule if not task.is_sleep_task()]
        prior_tasks = []

        # Check Each scheduledTask
        for i, task in enumerate(non_sleep_tasks):

            # Check Dependencies
            if not AbstractScheduler._verify_task_dependencies(task, prior_tasks):
                return False

            prior_tasks.append(task)

        return True

    '''Checks that every dependency for a task is scheduled prior to the task.

        '''
    @staticmethod
    def _verify_task_dependencies(task, prior_tasks):

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
    @staticmethod
    def _no_duplicate_tasks(tasklist: List[AbstractTask]):
        for task in tasklist:

            # ingore sleepTasks
            if task.is_sleep_task():
                continue

            if tasklist.count(task) > 1:
                return False

        return True

    @abstractmethod
    def schedule_tasks(self, tasklist: List[AbstractTask], interval: int) -> List[AbstractTask]:
        pass

    def simulate_execution(self, tasklist: List[AbstractTask], **kwargs):

        if "start_time" in kwargs:
            time = kwargs.get("start_time")
        elif self.start_time is None:
            time = datetime.now().timestamp()
        else:
            time = self.start_time.timestamp()

        total_value = 0

        for task in tasklist:

            # Ignore Sleep Tasks Values
            if not task.is_sleep_task():
                total_value += task.value(timestamp=time)

            time += task.wcet

        return total_value


class RandomScheduler(AbstractScheduler):

    def __init__(self, **kwargs):
        super().__init__()
        if "max_iterations" in kwargs:
            self.max_iteration = kwargs.get("max_iterations")
        else:
            self.max_iteration = 100

    def schedule_tasks(self, tasklist: List[AbstractTask], interval) -> List[AbstractTask]:
        new_tasklist = self._initialize_tasklist(tasklist, interval)

        valid = False
        schedule = None
        i = 0
        print("Generating Schedule at Random")
        while not valid:
            if self.verbose:
                print("Attempt: " + str(i + 1))
            schedule = random.sample(new_tasklist, len(new_tasklist))
            valid = self._validate_schedule(schedule)

            if valid:
                if self.verbose:
                    print("Valid Schedule Found")
                break
            elif i >= self.max_iteration:
                raise ValueError("Failed to generate a valid schedule after " + str(self.max_iteration) + " attempts")

            i += 1

        return schedule


class GeneticScheduler(AbstractScheduler):

    def __init__(self, **kwargs):
        super().__init__()
        if "max_generations" in kwargs:
            self.max_generations = kwargs.get("max_generations")
        else:
            self.max_generations = 500

        if "population_size" in kwargs:
            self.population_size = kwargs.get("population_size")
        else:
            self.population_size = 500
        self.elitism = True

        self._invalid_schedule_value = -1000.0
        self.breeding_percentage = .05
        self.mutation_rate = 0.01
        self.threshold = 0.01
        self.generation_thresold = 10
        self._tasks = None

    def schedule_tasks(self, tasklist: List[AbstractTask], interval) -> List[AbstractTask]:

        new_task_list = self._initialize_tasklist(tasklist, interval)
        population = []
        i = 1
        converged = False
        threshold_count = 0
        print("Generating Schedule Using Genetic Algorithm")

        # Initialize Population
        for x in range(self.population_size):
            population.append(random.sample(new_task_list, len(new_task_list)))

        current_best_schedule_value = 0

        while not converged:
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

            # Termination Conditions
            delta = abs(new_best_schedule_value - current_best_schedule_value)

            if i >= self.max_generations:
                break
            elif delta < self.threshold:
                threshold_count += 1

                if threshold_count >= self.generation_thresold:
                    converged = True

            else:
                threshold_count = 0

            print("Delta: " + str(delta) + " | thresold_count: " + str(threshold_count))
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
        if tasklist is None:
            tasklist = self._tasks

        if not AbstractScheduler._validate_schedule(schedule):
            return self._invalid_schedule_value
        elif not GeneticScheduler._all_tasks_present(tasklist, schedule):
            return self._invalid_schedule_value
        else:
            return self.simulate_execution(schedule)

    def _selection(self, population, **kwargs):
        values = []
        for schedule in population:
            values.append(self._fitness(schedule))

        sample = list(zip(population, values))
        sample.sort(key=lambda item: item[1], reverse=True)
        ordered_sample, _ = zip(*sample)
        cutoff = ceil(len(sample)*self.breeding_percentage)
        parent_sample = list(ordered_sample[:cutoff])

        return parent_sample

    ##
    # Creates the next generation of schudules. If "elitism is set", parents are carried over to next generation.
    def _crossover(self, parents, population_size):
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

        for schedule in population:
            for i, task in enumerate(schedule):
                if random.random() <= self.mutation_rate:
                    next_index = random.randint(0, len(schedule)-1)
                    temp = task
                    schedule[i] = schedule[next_index]
                    schedule[next_index] = temp

        return population

    @staticmethod
    def _all_tasks_present(original_schedule, new_schedule):
        for task in original_schedule:
            if task not in new_schedule:
                return False

        return True


class AntScheduler(AbstractScheduler):

    def __init__(self, **kwargs):
        super().__init__()

    def schedule_tasks(self, tasklist: List[AbstractTask], interval: int) -> List[AbstractTask]:
        pass


class Ant():

    def __init__(self):
        self._search_complete = False
        self._path = []

    def last_visited_node(self):
        if not self._path:
            return None
        else:
            return self._path[-1]

    def visit(self, node: AntTask, timestamp):
        self._path.append((node, timestamp))


class AntDependencyTree():

    def __init__(self, tasklist: List[AbstractTask]):
        self._nodes = []
        self._pheromones = {}
        self.population_size = 10000

        for task in tasklist:
            self._nodes.append(AntTask(task))

        self._update_dependencies()

    def get_node(self, task: AbstractTask):
        node = list(filter(lambda x: x._task is task, self._nodes))
        return node[0]

    def _update_dependencies(self):

        for node in self._nodes:
            dependencies = node._task.get_dependencies()

            for dependency in dependencies:
                found = False
                for other_node in self._nodes:
                    if dependency is other_node._task:
                        found = True
                        node.add_dependency(other_node)
                        break

                if not found:
                    raise ValueError("Dependency Error")

    def node_choices(self, ant: Ant):
        valid_choices = []

        # Remove nodes that have already been visited
        choices = [node for node in self._nodes if ant not in node.visited_by]

        # Only add nodes if their dependent nodes have already been visited
        for choice in choices:
            if all(map(lambda x: ant in x.visited_by,  choice.get_dependencies())):
                valid_choices.append(choice)

        # Deal With SleepTasks

        return valid_choices

    def visit_node(self, ant: Ant, node: AntTask, timestamp=None):
        '''

        :param ant:
        :param Node:
        :return:
        '''
        previous_node = ant.last_visited_node()

        node.visited_by.append(ant)


        # Update Pheromone Matrix
        edge = (ant.last_visited_node(), (node, timestamp))

        self._pheromones[edge] = self._pheromones.get(edge, 0) + 10000

        ant.visit(node, timestamp)

