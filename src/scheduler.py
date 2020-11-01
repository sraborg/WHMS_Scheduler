from abc import ABC, abstractmethod
import numpy as np
from task import AbstractTask, DummyTask, SleepTask, AntTask
from datetime import datetime
from typing import List, Dict, Tuple
from math import ceil
import random
import copy


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
    def simulated_annealing(cls):
        return SimulateAnnealingScheduler()

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


class AbstractScheduler(ABC):

    def __init__(self, **kwargs):
        self._tasks = None
        self._optimization_horizon = None
        self.start_time = kwargs.get("start_time", None)
        self.verbose = kwargs.get("verbose", False)
        self.invalid_schedule_value = kwargs.get("invalid_schedule_value", -1000.0)

    @property
    def optimization_horizon(self):
        if self._optimization_horizon is None:
            if self._tasks is None:
                raise ValueError("No tasks to schedule")
            else:
                self._optimization_horizon = AbstractScheduler.calculate_optimization_horizon(self._tasks)

        return self._optimization_horizon

    @staticmethod
    def calculate_optimization_horizon(tasklist):
        horizon = max(task.hard_deadline.timestamp() + task.wcet for task in tasklist)
        return horizon

    def _initialize_tasklist(self, tasklist: List[AbstractTask], interval):
        self._tasks = tasklist
        return tasklist + self.generate_sleep_tasks(tasklist, interval)

    def generate_sleep_tasks(self, tasklist: List[AbstractTask], interval, **kwargs):
        """
        optimization_horizon = None
        calculated_horizon = AbstractScheduler.calculate_optimization_horizon(tasklist)

        if self._optimization_horizon is None or calculated_horizon > self._optimization_horizon:
            optimization_horizon = calculated_horizon
        else:
            optimization_horizon = self._optimization_horizon
        """
        if "start_time" in kwargs:
            start_time = kwargs.get("start_time")
        else:
            start_time = datetime.now()

        # Calculate the number of sleep tasks needed
        total_wcet = sum(task.wcet for task in tasklist)
        dif = (self.optimization_horizon - start_time.timestamp()) - total_wcet
        num_sleep_tasks = ceil(dif / interval)

        # Generated Scheduled Sleep Tasks
        sleep_tasks = []
        for x in range(num_sleep_tasks):
            sleep_tasks.append(SleepTask(None, wcet=interval))

        return sleep_tasks

    @staticmethod
    def _validate_schedule(tasklist: List[AbstractTask]) -> bool:
        """Checks if Schedule is consistent with dependencies (e.g. no task is scheduled before any of its dependencies)

        """
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

    @staticmethod
    def simulate_step(task, time):
        value = 0

        if not task.is_sleep_task():
            value += task.value(timestamp=time)

        time += task.wcet

        return value, time


class MetaHeuristicScheduler(AbstractScheduler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_iterations = kwargs.get("max_iterations", 100)
        self.threshold = kwargs.get("threshold", 0.01)
        self.generational_threshold = kwargs.get("generational_threshold", 10)
        self._generational_threshold_count = 0

    def is_converged(self, last_value: float, current_value: float):
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


class GeneticScheduler(MetaHeuristicScheduler):
    """

    """
    def __init__(self, **kwargs):
        super().__init__()
        if "max_generations" in kwargs:
            self.max_generations = kwargs.get("max_generations")
        else:
            self.max_generations = 500

        if "population_size" in kwargs:
            self.population_size = kwargs.get("population_size")
        else:
            self.population_size = 5000
        self.elitism = True
        self.breeding_percentage = .05
        self.mutation_rate = 0.01
        self._tasks = None

    def schedule_tasks(self, tasklist: List[AbstractTask], interval) -> List[AbstractTask]:

        new_task_list = self._initialize_tasklist(tasklist, interval)
        population = []
        i = 1
        converged = False
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
            if i >= self.max_generations:
                break
            elif self.is_converged(current_best_schedule_value, new_best_schedule_value):
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

        if not AbstractScheduler._validate_schedule(schedule):
            return self.invalid_schedule_value
        elif not GeneticScheduler._all_tasks_present(tasklist, schedule):
            return self.invalid_schedule_value
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

    ##
    # Creates the next generation of schudules. If "elitism is set", parents are carried over to next generation.
    def _crossover(self, parents, population_size):
        """

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
        """

        :param population:
        :return:
        """
        for schedule in population:
            for i, task in enumerate(schedule):
                if random.random() <= self.mutation_rate:
                    next_index = random.randint(0, len(schedule)-1)
                    temp = task
                    schedule[i] = schedule[next_index]
                    schedule[next_index] = temp

        return population


class AntScheduler(MetaHeuristicScheduler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.colony_size = kwargs.get("colony_size", 15)
        self.alpha = kwargs.get("alpha", 1)
        self.beta = kwargs.get("beta", 1)
        self.epsilon = kwargs.get("epsilon", 0.5)

    def schedule_tasks(self, tasklist: List[AbstractTask], interval: int) -> List[AbstractTask]:
        """

        :param tasklist:
        :param interval:
        :return:
        """
        self._tasks = tasklist
        converged = False

        adt = AntDependencyTree(tasklist)
        possible_starting_nodes = adt.node_choices(Ant(), interval)
        i = 1
        converged = False
        self._generational_threshold_count = 0
        possible_solutions: List[Tuple[List[AbstractTask], int]] = [([], 0)]  # List of (Schedule, Value)

        while not converged:
            print("Processing Swarm " + str(i) + " of " + str(self.colony_size) + " ants")

            # Initialization
            colony = []

            for ant in range(self.colony_size):
                colony.append(Ant())

            for ant in colony:

                # Place ants on random "Starting Node" (e.g. the "top" of the graph).
                node = random.choice(possible_starting_nodes)
                time = self.start_time.timestamp()

                adt.visit_node(ant, node, time)
                step = 1

                # Generate Path for each ant
                while not ant._search_complete:

                    # Determine which nodes the ant can visit
                    valid_choices = adt.node_choices(ant, interval)

                    # Check for Path Termination (e.g. empty list or exceeded event horizon)
                    if not valid_choices:
                        ant._search_complete = True
                        break
                    elif ant._time >= self.optimization_horizon:
                        break

                    # Make move
                    next_node = self._edge_selection(ant, valid_choices, adt, interval+1)
                    time += node.wcet
                    adt.visit_node(ant, next_node, time)

                    step += 1

            # Update Best Solutions
            new_solutions = []
            for ant in colony:
                sch = ant.get_schedule()
                val = self.simulate_execution(sch)
                new_solutions.append((sch, val))

            current_best_schedule_value = possible_solutions[0][1]
            possible_solutions = possible_solutions + new_solutions
            possible_solutions.sort(key=lambda x: x[1], reverse=True)
            possible_solutions = possible_solutions[:ceil(len(colony)/2)]

            new_best_schedule_value = possible_solutions[0][1]
            if self.verbose:
                print("Best Path ("+str(len(possible_solutions[0][0])) + "): " + str(new_best_schedule_value))

            # Algorithm Termination Conditions
            if self.is_converged(current_best_schedule_value, new_best_schedule_value):
                print("Convergence Met")
                converged = True
                break
            elif i >= self.max_iterations:
                print("Max iterations met" + str(i) + " | " + str(self.max_iterations))
                break

            # Prep Next iteration
            i += 1

            # Update Pheromone Matrix
            for ant in colony:
                adt.update_pheromones(ant, self._fitness)

            # Evaporation
            adt.evaporate_pheromones()

        print("Finished after " + str(i) + " swarms")

        best_schedule = list(possible_solutions[0][0])
        return best_schedule

    def _edge_selection(self, ant, choices, adt, iteration=1):
        """ Choices then next node to visit (and hence the edge to use).

        :param ant:
        :param choices: List of valid choices
        :return:
        """

        if random.uniform(0,1) < self.epsilon / iteration:
            return choices[random.randint(0, len(choices) - 1)]
        else:
            probabilities = []
            last = ant.last_visited_node()
            alpha = self.alpha
            beta = self.beta / iteration
            for choice in choices:
                e = (last, choice)
                p_value = adt.get_pheromone_value(e)
                if p_value < 1:
                    p_value = -1*p_value
                    p_value = -1*(p_value**alpha)
                else:
                    p_value = p_value ** alpha

                h_value = self._attractiveness(choice, last[1])
                if h_value < 1:
                    h_value = -1 * h_value
                    h_value = -1 * (h_value ** beta)
                else:
                    h_value = h_value ** beta

                value = p_value * h_value


                probabilities.append(value)

            # Shift probabilities to remove negative values
            smallest = min(probabilities)
            if smallest < 0:
                probabilities = [p - smallest for p in probabilities]

            # Normalize
            norm = sum(probabilities)
            probabilities = [p/norm for p in probabilities]

            # Make a weighted Selection
            node_choice = random.choices(choices, weights=probabilities)
            return node_choice[0]

    def _attractiveness(self, Node, time):
        """ Heuristic function that takes into consideration:
        (1) the amount of value lost if the task is taken early

        :param Node:
        :param time:
        :return:
        """
        if Node.is_sleep_task():
            return 0

        value_now = Node.value(timestamp=time)
        utopia_point = Node.soft_deadline.timestamp()

        if time > utopia_point:
            return value_now
        else:

            utopia_value = Node.value(timestamp=utopia_point)
            return value_now - utopia_value

    def _fitness(self, schedule):
        """ Provides a value for the given schedule. For invalid schedules, the "invalid schedule value" is returned.
        Otherwise, the simulated value is returned.

        :param schedule:
        :return:
        """
        if not AbstractScheduler._validate_schedule(schedule):
            return self.invalid_schedule_value
        else:
            return self.simulate_execution(schedule)


class Ant:

    def __init__(self):
        self._search_complete = False
        self._path = []                 # [(AntTask, timestamp)]
        self._ant_tasks = []            # [(AntTask)]
        self._schedule = []             # [AbstractTasks]
        self._path_value = None
        self._time = 0

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
        return list(map(lambda x: x[0]._task, self._path))

    def last_visited_node(self):
        visited_nodes = list(self.get_visited_nodes())
        if not visited_nodes: # empty case
            return None

        return visited_nodes[-1]

    def visit(self, node: AntTask, timestamp):
        """

        :param node:
        :param timestamp:
        :return:
        """
        self._time = timestamp
        self._path.append((node, timestamp))
        self._ant_tasks.append(node)
        self._schedule.append(node._task)

    def get_path_value(self):
        """

        :return:
        """
        if not self._search_complete:
            raise ValueError("Ant has not completed path")

        if self._path_value is None:
            self._path_value = 0


class AntDependencyTree:

    def __init__(self, tasklist: List[AbstractTask]):
        self._nodes = []
        self._pheromones: Dict[List[float]] = {}
        self._edge_visits: Dict[Tuple[int,int]] = {}
        self.min_pheromone_amount = 0.001

        for task in tasklist:
            self._nodes.append(AntTask(task))

        self._update_dependencies()

    def get_ant_task(self, task: AbstractTask):
        node = list(filter(lambda x: x._task is task, self._nodes))
        return node[0]

    def _update_dependencies(self):
        """Fixes the dependency references so that to point to the Ant Task Wrapper
        instead of the owned (composite) task class

        :return: None
        """

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

    def node_choices(self, ant: Ant, interval: float):
        """ Determines which nodes are available to the ant. In other words, it only excludes nodes (tasks) whose
            dependencies haven't been met or if it has already been visited by the ant.

        :param ant: Ant
        :param interval: Only Used to dynamically create SleepTasks as needed
        :return: List(AntTasks)
        """
        valid_choices = []
        completed_ant_tasks = list(ant.get_completed_ant_tasks())

        # Remove nodes that have already been visited
        choices = [task for task in self._nodes if task not in completed_ant_tasks]

        # Only add nodes if their dependent nodes have already been visited
        for choice in choices:
            if all(map(lambda x: ant in x.visited_by,  choice.get_dependencies())):
                valid_choices.append(choice)

        # If there are still nodes to visit, add SleepTask
        if not not valid_choices:
            st = SleepTask(runtime=interval, analysis_type="SLEEPANALYSIS", wcet=interval)
            valid_choices.append(AntTask(st))

        return valid_choices

    def visit_node(self, ant: Ant, node: AntTask, timestamp=None):
        """
        :param ant:
        :param node:
        :param timestamp:
        :return:
        """

        previous_node = ant.last_visited_node()

        node.accept(ant, timestamp)



        # Update Pheromone Matrix
        # edge = (ant.last_visited_node(), (node, timestamp))

        #self._pheromones[edge] = self._pheromones.get(edge, 0) + 10000


    def update_pheromones(self, ant, fitness):
        """

        :param ant: the Ant
        :param fitness: A Fitness Function
        :return:
        """
        schedule = list(ant.get_schedule())

        value = fitness(schedule)

        path = list(ant.get_visited_nodes())
        for i, node in enumerate(path[:-1]):
            key = (path[i], path[i+1])

            if key not in self._pheromones:
                self._pheromones[key] = []

            if key not in self._edge_visits:
                self._edge_visits[key] = 0

            self._pheromones[key].append(value)
            self._edge_visits[key] = self._edge_visits[key] + 1

    def get_pheromone_value(self, key):
        if key not in self._pheromones:
            return self.min_pheromone_amount
        else:
            return sum(self._pheromones[key]) / self._edge_visits[key]

    def evaporate_pheromones(self, evaporation_rate=0.8):
        for k,v in self._pheromones.items():
            self._pheromones[k] = list(map(lambda x: x * evaporation_rate, v))


class SimulateAnnealingScheduler(MetaHeuristicScheduler):
    def schedule_tasks(self, tasklist: List[AbstractTask], interval: int) -> List[AbstractTask]:
        pass

