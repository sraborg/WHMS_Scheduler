from abc import ABC, abstractmethod
from operator import itemgetter
import numpy as np
from task import AbstractTask, SleepTask, AntTask, UserTask
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from math import ceil, floor, e
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
        self._cached_schedule_values: Dict[float] = {}
        self.end_time: datetime = kwargs.get("end_time", None)
        self.verbose = kwargs.get("verbose", False)
        self.invalid_schedule_value = kwargs.get("invalid_schedule_value", -1000.0)
        self._utopian_schedule_value = None
        self._flag_generate_sleep_tasks = True
        self._flag_generate_periodic_tasks = True

    @property
    def optimization_horizon(self):
        """ Accessor method for the optimization_horizon attribute.

        Note if the scheduler's end_time attribute is not set, an optimization_horizon will be calculated (based off the tasklist)
        and the end_time will be set.

        :return:
        """

        # Lazy-Load
        if self._optimization_horizon is None:

            # Simple case
            if self.end_time is not None:
                self._optimization_horizon = self.end_time - self.start_time

            # Missing End_time case (Generate an optimization horizon & end_time)
            else:
                if self._tasks is None:
                    raise ValueError("No tasks to schedule")
                else:
                    self._optimization_horizon = AbstractScheduler.calculate_optimization_horizon(self, self._tasks)
                    self.end_time = self.start_time + self._optimization_horizon

        return self._optimization_horizon

    @optimization_horizon.setter
    def optimization_horizon(self, value):
        self._optimization_horizon = value

    def calculate_optimization_horizon(self, tasklist):
        """

        :param tasklist:
        :return: the generated horizon as a timedelta
        """
        temp = [task.hard_deadline for task in tasklist]
        latest_task = max([task.hard_deadline.timestamp() + task.wcet for task in tasklist])
        horizon = datetime.fromtimestamp(latest_task) - self.start_time
        return horizon

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

    def generate_sleep_tasks(self, tasklist: List[AbstractTask], interval, **kwargs):
        """

        :param tasklist:
        :param interval:
        :param kwargs:
        :return:
        """

        """
        optimization_horizon = None
        calculated_horizon = AbstractScheduler.calculate_optimization_horizon(tasklist)

        if self._optimization_horizon is None or calculated_horizon > self._optimization_horizon:
            optimization_horizon = calculated_horizon
        else:
            optimization_horizon = self._optimization_horizon

        if "start_time" in kwargs:
            start_time = kwargs.get("start_time")
        else:
            start_time = self.start_time
        """

        # Calculate the number of sleep tasks needed
        total_wcet = sum(task.wcet for task in tasklist)
        dif = self.optimization_horizon.total_seconds() - total_wcet
        num_sleep_tasks = ceil(dif / interval)

        # Generated Scheduled Sleep Tasks
        sleep_tasks = []
        for x in range(num_sleep_tasks):
            sleep_tasks.append(SleepTask(wcet=interval))

        return sleep_tasks

    def generate_periodic_tasks(self, tasklist: List[AbstractTask], **kwargs):
        """

        :param tasklist:
        :param interval:
        :param kwargs:
        :return:
        """
        new_periodic_tasks = []

        for task in tasklist:
            num_periodic_tasks: int = 0

            # Find Periodic tasks
            if task.periodicity > 0:

                # Determine How many future periodic tasks are possible
                horizon = self.end_time - task.earliest_start
                max_num_periodic_tasks: int = floor(horizon.total_seconds() / task.periodicity)

                # Generate New Periodic Tasks
                for i in range(max_num_periodic_tasks):

                    # interval shift
                    shift: float = (i+1) * task.periodicity

                    new_task: AbstractTask = copy.deepcopy(task)
                    new_task.periodicity = -1

                    new_task.nu.shift_deadlines(shift)

                    new_periodic_tasks.append(new_task)

        return new_periodic_tasks


    @staticmethod
    def _validate_schedule(tasklist: List[AbstractTask]) -> bool:
        """ Checks if Schedule is consistent with dependencies (e.g. no task is scheduled before any of its dependencies)

        :param tasklist:
        :return:
        """

        # Check for Duplicates
        if not AbstractScheduler._no_duplicate_tasks(tasklist):
            return False

        # Check Dependencies
        if not AbstractScheduler._verify_schedule_dependencies(tasklist):
            return False

        return True

    @staticmethod
    def _verify_schedule_dependencies(schedule: List[AbstractTask]):
        """ Checks that every dependency for every task is scheduled prior to the task

        :param schedule:
        :return:
        """
        non_sleep_tasks = [task for task in schedule if not task.is_sleep_task()]
        prior_tasks = []

        # Check Each scheduledTask
        for i, task in enumerate(non_sleep_tasks):

            # Check Dependencies
            if not AbstractScheduler._verify_task_dependencies(task, prior_tasks):
                return False

            prior_tasks.append(task)

        return True

    @staticmethod
    def _verify_task_dependencies(task, prior_tasks):
        """ Checks that every dependency for a task is scheduled prior to the task.

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
    def _no_duplicate_tasks(tasklist: List[AbstractTask]) -> bool:
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
        """Simulates executes of a schedule

        :param tasklist:
        :param start:
        :param kwargs:
        :return: the schedule's value
        """
        if start is None:
            time = self.start_time.timestamp()
        else:
            time = start

        key = (tuple(tasklist), start)
        if key in self._cached_schedule_values.keys():
            return self._cached_schedule_values[key]
        else:

            total_value = 0

            for task in tasklist:

                # Ignore Sleep Tasks Values
                if not task.is_sleep_task():
                    total_value += task.value(timestamp=time)

                time += task.wcet

            self._cached_schedule_values[key] = total_value

        return self._cached_schedule_values[key]

    """
    @staticmethod
    def simulate_step(task, time):
        value = 0

        if not task.is_sleep_task():
            value += task.value(timestamp=time)

        time += task.wcet

        return value, time
    """
    @staticmethod
    def utopian_schedule_value(schedule):
        """ Gets the Utopian (the best) value. Note this value may not be achievable.

        :param schedule:
        :return:
        """

        value = 0
        for task in schedule:
            date_time = task.soft_deadline
            time = date_time.timestamp()
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

        utopian_value = AbstractScheduler.utopian_schedule_value(schedule)
        weighted_value = value / utopian_value

        return weighted_value


class MetaHeuristicScheduler(AbstractScheduler):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.duration = kwargs.get("duration", 1) # Duration Stored in minutes
        self.max_iterations = kwargs.get("max_iterations", 100)
        self.threshold = kwargs.get("threshold", 0.01)
        self.generational_threshold = kwargs.get("generational_threshold", 10)
        self._generational_threshold_count = 0
        self._last_value: float = 0
        self._current_value: float = 0
        self._progress = 0

        self._flag_termination_by_duration = True
        self._flag_termination_by_max_iteration = False
        self._flag_termination_by_generational_delta = False

    def _termination_by_duration(self, start_runtime: datetime) -> bool:
        """ Checks whether an the algorithm has exceeded it's allowed duration.

        :param start_runtime: The time the algorithm started
        :return: True/False
        """
        elapsed_time = datetime.now() - start_runtime

        progress: int = ceil((elapsed_time.total_seconds() / (self.duration * 60)*100))
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

    ANT_SYSTEM = 0
    ANT_COLONY = 1

    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super().__init__(**kwargs)
        self.colony_size = kwargs.get("colony_size", 15)
        self.alpha = kwargs.get("alpha", 1)
        self.beta = kwargs.get("beta", 1)
        self.epsilon = kwargs.get("epsilon", 0.5)
        self.pheromone_update_method = self.ANT_SYSTEM
        self._flag_generate_sleep_tasks = False
        self._flag_aco_pheromone_update = True
        self._flag_local_pheromone_update = False

    def schedule_tasks(self, tasklist: List[AbstractTask], interval: int) -> List[AbstractTask]:
        """ Generates a schedule using a modified ant colony optimization algorithm

        :param tasklist: the list of tasks to be scheduled
        :param interval: the period of time to divide the optimization horizon
        :return: an ordered list of tasks
        """
        print("Generating Schedule Using Ant Colony Algorithm")
        start_runtime = datetime.now()
        new_task_list = self._initialize_tasklist(tasklist, interval)

        adt = AntDependencyTree(new_task_list, min_max=True)

        root_node = AntTask(SleepTask(runtime=0, analysis_type="SLEEPANALYSIS", wcet=0))

        # Get Valid starting nodes
        #possible_starting_task = adt.ant_task_choices(Ant(), interval)
        i = 1
        converged = False
        self._generational_threshold_count = 0

        global_solutions: List[Tuple[List[AbstractTask], int]] = [([], 0)]  # List of (Schedule, Value)

        while not converged:
            if self.verbose:
                print("Processing Swarm " + str(i) + " of " + str(self.colony_size) + " ants")

            # Generate Ant Swarm
            colony = []
            for ant in range(self.colony_size):
                colony.append(Ant())

            # Send each Ant to explore
            for ant in colony:

                # Place ants on random "Starting Node" (e.g. the "top" of the graph).
                ant_task = root_node #random.choice(possible_starting_task)
                time = self.start_time.timestamp()

                adt.visit_node(ant, ant_task, time)
                step = 1

                # Generate Path for each ant
                while not ant._search_complete:

                    # Determine which nodes the ant can visit
                    valid_choices = adt.ant_task_choices(ant, interval)

                    # Check for Path Termination (e.g. empty list or exceeded horizon)
                    if not valid_choices:
                        ant._search_complete = True
                        break
                    # Check if the ant has taken too much time
                    elif ant.simulated_time >= self.end_time.timestamp():
                        break

                    # Make move
                    time += ant_task.wcet
                    ant_task = self._edge_selection(ant, valid_choices, adt, interval+1)

                    adt.visit_node(ant, ant_task, time)

                    step += 1

                # Termination by Duration - Stop the rest of the swarm from searching
                if self._flag_termination_by_duration and self._termination_by_duration(start_runtime):
                    break



            # Termination by Duration
            if self._flag_termination_by_duration and self._termination_by_duration(start_runtime):
                break

            # Termination by Max Iteration
            if self._flag_termination_by_max_iteration and self._termination_by_max_iterations(i):
                print("Max iterations met" + str(i) + " | " + str(self.max_iterations))
                break

            '''
            # Termination by Generational Delta
            if self._flag_termination_by_generational_delta and \
                    self._termination_by_generational_delta(current_best_schedule_value, new_best_schedule_value):
                print("Generational Delta Threshold Met")
                print("Convergence Met after " + str(i) + " iterations")
                break
            '''

            # Update Best Solutions
            new_solutions = []
            for ant in colony:
                sch = ant.get_schedule()
                val = self.simulate_execution(sch)
                new_solutions.append((sch, val))

            current_best_schedule_value = global_solutions[0][1]
            global_solutions = global_solutions + new_solutions
            global_solutions.sort(key=lambda x: x[1], reverse=True)
            global_solutions = global_solutions[:ceil(len(colony) / 2)]

            new_best_schedule_value = global_solutions[0][1]
            if self.verbose:
                print("Best Path (" + str(len(global_solutions[0][0])) + "): " + str(new_best_schedule_value))

            # Prep Next iteration
            i += 1

            # Update Pheromone Matrix
            if self.pheromone_update_method == self.ANT_SYSTEM:
                adt.update_pheromones(colony, self._fitness)
                """
                for ant in colony:
                    adt.update_pheromones(ant, self._fitness)
                """
            elif self.pheromone_update_method == self.ANT_COLONY:
                pass

            # Evaporation
            #adt.evaporate_pheromones()

        print("Finished after " + str(i) + " swarms")

        best_schedule = list(global_solutions[0][0])
        sat_path = best_schedule = list(global_solutions[0][0]) #adt.get_saturated_path(root_node, self.start_time, self.end_time, interval)

        return sat_path[1:] # Skip the root_node best_schedule

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
            alpha = self.alpha + iteration/10
            beta = self.beta / iteration
            for choice in choices:
                e = (last, choice)

                # Get Pheromone Value
                p_value = adt.get_pheromone_value(e)

                '''
                if p_value < 1:
                    p_value = 1
                    """
                    p_value = -1*p_value
                    p_value = -1*(p_value**alpha)
                    """
                else:
                    p_value = p_value ** alpha
                '''

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
       #hard_deadline = Node.hard_deadline.timestamp()

        if time < utopia_point:
            utopia_value = Node.value(timestamp=utopia_point)
            return value_now - (utopia_value - value_now)
        else:
            return value_now

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
        return list(map(lambda x: x[0]._task, self._path))

    def last_visited_node(self):
        visited_nodes = list(self.get_visited_nodes())
        if not visited_nodes: # empty case
            return None

        return visited_nodes[-1]

    def visit(self, ant_task: AntTask, timestamp):
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

    def ant_task_choices(self, ant: Ant, interval: float):
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
            st = SleepTask(runtime=interval, analysis_type="SLEEPANALYSIS", wcet=interval)
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

    def update_pheromones(self, ants: List[Ant], fitness):
        """

        :param ant: the Ant
        :param fitness: A Fitness Function
        :return:
        """
        new_pheromones: Dict[float] = {}

        # Calculate new pheromones to add
        for ant in ants:
            schedule = list(ant.get_schedule())

            value = fitness(schedule)

            path = list(ant.get_visited_nodes())
            for i, node in enumerate(path[:-1]):
                key = (path[i], path[i+1])

                current = new_pheromones.get(key, self.initial_pheromone_amount)
                new_pheromones[key] = current + value

        for key, value in new_pheromones.items():
            current = self._pheromomnes.get(key, 0)
            self._pheromomnes[key] = (1-self._evaporation_rate) + value

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

class HeuristicScheduler(AbstractScheduler):

    def schedule_tasks(self, tasklist: List[AbstractTask], interval: int) -> List[AbstractTask]:
        pass


class SimulateAnnealingScheduler(MetaHeuristicScheduler):

    linear_decay = 0
    geometric_decay = 1
    slow_decay = 2

    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super().__init__(**kwargs)
        self.temperature = kwargs.get("temperature", 10000)
        self.solution_space_size = kwargs.get("solution_space_size", 10000)
        self.decay_method = kwargs.get("decay_method", SimulateAnnealingScheduler.geometric_decay)
        self.decay_constant = kwargs.get("decay_constant", 0.9)
        self.elitism = True
        self.num_neighbors = 2

    def schedule_tasks(self, tasklist: List[AbstractTask], interval: int) -> List[AbstractTask]:
        print("Generating Schedule with Simulated Annealing")
        start_runtime = datetime.now()

        new_task_list = self._initialize_tasklist(tasklist, interval)

        current_state = random.sample(new_task_list, len(new_task_list))
        current_state_energy = self._energy(current_state)
        best_solution = current_state
        best_solution_value = self.simulate_execution(best_solution)

        converged = False
        i = 0
        temp = self.temperature
        while not converged:
            temp = self._temperature(temp, self.decay_method)

            neighbor = self._neighbors(current_state)
            neighbor_energy = self._energy(neighbor)

            # Possibly change state
            if self._probability(current_state_energy, neighbor_energy, temp) >= random.uniform(0, 1):
                current_state = neighbor
                current_state_energy = neighbor_energy

                if self.elitism:
                    old_best_value = best_solution_value
                    current_state_value = self.simulate_execution(current_state)

                    if best_solution_value < current_state_value:
                        best_solution = current_state
                        best_solution_value = current_state_value

                    # Termination by Generational Delta
                    if self._flag_termination_by_generational_delta and \
                            self._termination_by_generational_delta(old_best_value, best_solution_value): # Checks delta
                        print("Generational Delta Threshold Met")
                        print("Convergence Met after " + str(i) + " iterations")
                        break
                else:
                    pass

            # Termination by Max Iteration
            if self._flag_termination_by_max_iteration and self._termination_by_max_iterations(i):
                print("Max iterations met" + str(i) + " | " + str(self.max_iterations))
                break

            i += 1

            # Termination by Duration
            if self._flag_termination_by_duration and self._termination_by_duration(start_runtime):
                 break

        if self.elitism:
            return best_solution
        else:
            return current_state

    def _temperature(self, temp, decay_method=slow_decay):
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

    def _neighbors(self, state):
        """

        :param state:
        :param solution_space:
        :return:
        """

        if random.uniform(0, 1) < 0.5:
            return self._reverse(state)
        else:
            return self._transport(state)

    def _reverse(self, state):
        """

        :param state:
        :return:
        """
        new_state = copy.copy(state)
        index = random.randint(0, len(new_state) - 1)
        next_index = random.randint(0, len(new_state) - 1)
        temp = new_state[index]
        new_state[index] = new_state[next_index]
        new_state[next_index] = temp

        return new_state

    def _transport(self, state):
        new_state = copy.copy(state)
        index = random.randint(0, len(new_state) - 1)
        next_index = random.randint(index, len(new_state) - 1)
        slice = state[index:next_index]

        del new_state[index:next_index]

        insertion_index = random.randint(0, len(new_state) - 1)
        new_state[insertion_index:insertion_index] = slice

        return new_state

    def _probability(self, energy, energy_new, temp):
        change = energy - energy_new
        if change > 0:
            return 1
        else:
            return e**(-change/temp)

    def _energy(self, state):
        return 1 / self.simulate_execution(state)