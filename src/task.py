from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, List
import json
import copy           # Used in task_builder to fix error with getTask
from nu import NuFactory, NuRegression, NuConstant
from analysis import AnalysisFactory
from analysis import SleepAnalysis
from nu import NuRegression
import random


class AbstractTask(ABC):

    def __init__(self, **kwargs):
        self.analysis = kwargs.get("analysis", None)
        self.values = kwargs.get("values", [])
        self.nu = kwargs.get("nu", None)
        self.periodicity: timedelta = timedelta(seconds=kwargs.get("periodicity", 0))
        self._cost = kwargs.get("cost", None)
        self._dependent_tasks: List[AbstractTask] = kwargs.get("dependent_tasks", [])
        self._dynamic_tasks = kwargs.get("dynamic_tasks", [])        # potential tasks
        self._future_tasks = []                           # Tasks


    @property
    def earliest_start(self):
        return self.nu.earliest_start

    @property
    def soft_deadline(self):
        return self.nu.soft_deadline

    @property
    def hard_deadline(self):
        return self.nu.hard_deadline

    @property
    def utopian_value(self):
        return self.nu.utopian_value()

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, cost):
        self._cost = cost

    @property
    def dependent_tasks(self):
        return self._dependent_tasks

    @dependent_tasks.setter
    def dependent_tasks(self, tasks):
        self._dependent_tasks = tasks

    @property
    def dynamic_tasks(self):
        return self._dynamic_tasks

    @dynamic_tasks.setter
    def dynamic_tasks(self, tasks):
        self._dynamic_tasks = tasks

    @property
    def future_tasks(self):
        return self._future_tasks

    @future_tasks.setter
    def future_tasks(self, tasks):
        self._future_tasks = tasks

    @property
    def wcet(self):
        return self.analysis.wcet

    @property
    def wcbu(self):
        return self.analysis.wcbu

    def execute(self):
        self.analysis.execute()

    def value(self, **kwargs):
        if "timestamp" in kwargs:
            timestamp = kwargs.get("timestamp").timestamp()
        else:
            timestamp = datetime.now().timestamp()
        return self.nu.eval(timestamp)

    def has_dependencies(self):
        return len(self._dependent_tasks) > 0

    def get_dependencies(self):
        return copy.copy(self._dependent_tasks)

    def is_dependency(self, task):
        result = False

        for dependency in self._dependent_tasks:
            if task is dependency:
                result = True
                break
        return result

    def add_dependency(self, dependency):
        self._dependent_tasks.append(dependency)

    def remove_dependency(self, dependency):
        """Removes a task from dependency list if present. Otherwise, does nothing

        """
        if self.is_dependency(dependency):
            self._dependent_tasks.remove(dependency)
        else:
            return

    def is_dummy(self):
        return isinstance(self, DummyTask)

    def is_sleep_task(self):
        return isinstance(self.analysis, SleepTask)

    def is_periodic(self):
        return False

    @abstractmethod
    def name(self):
        pass

    @staticmethod
    def load_tasks(path):
        tasks = []
        #tb = TaskBuilder()
        with open(path, 'r') as json_file:
            loaded_tasks = json.load(json_file)

            for entry in loaded_tasks:

                task = UserTask()

                # Setup Analysis
                task.analysis = AnalysisFactory.get_analysis(entry["analysis"])

                #task.earliest_start = datetime.fromtimestamp(float(entry["earliest_start"]))
                #task.soft_deadline = datetime.fromtimestamp(float(entry["soft_deadline"]))
                #task.hard_deadline = datetime.fromtimestamp(float(entry["hard_deadline"]))
                task.ordered_by = entry["ordered_by"]
                #task.values = entry["values"]
                task.periodicity = timedelta(seconds=entry["periodicity"])

                # Setup Nu
                nu = NuFactory.get_nu(entry["nu"])
                nu.fit_model(list(map(lambda x: tuple(x), entry["values"])))
                task.nu = nu

                # Handle Dependencies
                dependent_task_indices = entry["dependent_tasks"].split(";")
                for dependency in dependent_task_indices:
                    if len(dependency) > 0:
                        index = int(dependency)
                        task.add_dependency(tasks[index])

                tasks.append(task)

        return Schedule(tasks)

    @staticmethod
    def save_tasks(path, tasklist):
        with open(path, mode='w') as json_file:

            entries = []

            for i, task in enumerate(tasklist):

                # Determine Dependencies
                dependent = ''
                dependent_delim = ""
                for dependency in task.dependent_tasks:
                    dependent += dependent_delim + str(tasklist.index(dependency))
                    dependent_delim = ";"

                entry = {
                    'analysis': task.analysis.name(),
                    'earliest_start': task.earliest_start.timestamp(),
                    'soft_deadline': task.soft_deadline.timestamp(),
                    'hard_deadline': task.hard_deadline.timestamp(),
                    'ordered_by': task.ordered_by,
                    'dependent_tasks': dependent,
                    'nu': task.nu.name(),
                    'values': task.values,
                    'periodicity': task.periodicity,
                }

                # Determine Values
                if isinstance(task.nu, NuRegression):
                    pass
                elif isinstance(task.nu, NuConstant):
                    entry['value'] = task.nu.eval()

                entries.append(entry)
            json.dump(entries, json_file)


    @staticmethod
    def generate_random_tasks(quantity=10, dependencies=True, start: datetime = None, end:datetime=None, max_value:int=1000):
        tasklist = []
        random.seed()

        if start is None:
            start = datetime.now()
        if end is None:
            end = start + timedelta(minutes=60)

        diff = end-start

        for i in range(quantity):

            t = UserTask()

            # Setup Analysis
            analysis_types = [
                AnalysisFactory.body_temperature_analysis,
                AnalysisFactory.blood_glucose_analysis,
                AnalysisFactory.blood_pressure_analysis,
                AnalysisFactory.electrocardiogram_analysis,
                AnalysisFactory.electroencephalography_analysis,
                AnalysisFactory.exercise_analysis,
                AnalysisFactory.heart_rate_analysis,
                AnalysisFactory.motion_analysis,
                AnalysisFactory.respiratory_rate_analysis,
                AnalysisFactory.stress_analysis,
            ]
            t.analysis = random.choice(analysis_types)()

            names = [
                "Dr Patterson",
                "Dr Cortez",
                "Dr Holland",
                "Dr Page",
                "Dr Boyd",
                "Dr Zuniga",
                "Dr Robinson"
            ]

            t.ordered_by = random.choice(names)

            earliest_start = datetime.fromtimestamp(random.uniform(start.timestamp(),end.timestamp()))
            soft_deadline = datetime.fromtimestamp(random.uniform(earliest_start.timestamp(), end.timestamp()))
            hard_deadline = datetime.fromtimestamp(random.uniform(soft_deadline.timestamp(), end.timestamp()))

            '''
            t.earliest_start = earliest_start
            t.soft_deadline = soft_deadline
            t.hard_deadline = hard_deadline
            '''

            t.values = [
                (earliest_start.timestamp(), 0),
                (soft_deadline.timestamp(), random.randint(0, max_value)),
                (hard_deadline.timestamp(), 0)
            ]

            """
            # Random Start Time Between Start & (End - 10 min)
            delta = random.uniform(0, (diff.total_seconds() / 60) - 10)
            earliest_start = start + timedelta(minutes=delta)

            # Random Soft Deadline Between (Earliest + 1 min) & (End - 5 min)
            earliest_start_diff = earliest_start - end
            delta = random.uniform(1, (earliest_start_diff.total_seconds() / 60) - 5)
            soft_deadline = earliest_start + timedelta(minutes=delta) #timedelta(minutes=random.randint(0, 10), seconds=random.randint(0, 30))

            # Random Hard Deadline Between (Soft + 1 & End)
            soft_deadline_diff = soft_deadline - end
            delta = random.uniform(1, soft_deadline_diff.total_seconds() / 60)
            hard_deadline = soft_deadline + timedelta(minutes=delta) #timedelta(minutes=random.randint(0, 10), seconds=random.randint(0, 30))

            t = UserTask()
            t.earliest_start = earliest_start
            t.soft_deadline = soft_deadline
            t.hard_deadline = hard_deadline
            t.values = [
                (earliest_start.timestamp(), 0),
                (soft_deadline.timestamp(), random.randint(0, max_value)),
                (hard_deadline.timestamp(), 0)
            ]
            """

            nu = NuFactory.regression()
            nu.fit_model(t.values)
            t.nu = nu

            # Random Chance to assign Dependencies
            if dependencies and i > 0 and random.randint(0, 1) == 1:
                t.add_dependency(tasklist[i - 1])
            tasklist.append(t)

        return tasklist


class UserTask(AbstractTask):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.ordered_by = kwargs.get("ordered_by", "")
        self._queue_time = None
        self._release_time: datetime = None
        self._completion_time: datetime = None
        self._execution_time: datetime = None

    def name(self):
        return "User"

    @property
    def queue_time(self):
        return self._queue_time

    @queue_time.setter
    def queue_time(self, time):
        self._queue_time = time

    @property
    def release_time(self):
        return self._release_time

    @release_time.setter
    def release_time(self, time):
        self._release_time = time

    @property
    def completion_time(self):
        return self._completion_time

    @completion_time.setter
    def completion_time(self, time):
        self._completion_time = time

    @property
    def execution_time(self):
        return self._execution_time

    @execution_time.setter
    def execution_time(self, time):
        self._completion_time = time


class SystemTask(AbstractTask):
    pass


class SleepTask(SystemTask):

    def __init__(self, **kwargs):
        super().__init__()

        # Setup NuConstant
        value = kwargs.get("SLEEP_VALUE", 0)
        self.nu = NuConstant(CONSTANT_VALUE=value)

        # Setup SleepAnalysis
        wcet = kwargs.get("wcet", timedelta(5))
        self.analysis = SleepAnalysis(wcet=wcet)

    def execute(self):
        self.analysis.execute()

    def name(self):
        return "SLEEP"


class TaskDecorator(ABC):

    def __init__(self, task: AbstractTask):
        super().__init__()
        self._task = task

    def value(self, **kwargs):
        return self._task.value(**kwargs)

    @property
    def cost(self):
        return self._task.cost

    @cost.setter
    def cost(self, cost):
        self._task.cost = cost

    @property
    def nu(self):
        return self._task.nu

    @nu.setter
    def nu(self, method):
        self._task.nu = method

    @property
    def analysis(self):
        return self._task.analysis

    @analysis.setter
    def analysis(self, analysis_type):
        self._task.analysis = analysis_type

    @property
    def earliest_start(self):
        return self._task.earliest_start

    @earliest_start.setter
    def earliest_start(self, deadline):
        self._task.earliest_start = deadline
    @property
    def soft_deadline(self):
        return self._task.soft_deadline

    @soft_deadline.setter
    def soft_deadline(self, deadline):
        self._task.soft_deadline = deadline

    @property
    def hard_deadline(self):
        return self._task.hard_deadline

    @hard_deadline.setter
    def hard_deadline(self, deadline):
        self._task.hard_deadline = deadline

    @property
    def utopian_value(self):
        return self._task.utopian_value()

    @property
    def ordered_by(self):
        return self._task.ordered_by

    @ordered_by.setter
    def ordered_by(self, value):
        self._task.ordered_by = value

    @property
    def dependent_tasks(self):
        return self._task.dependent_tasks

    @dependent_tasks.setter
    def dependent_tasks(self, tasks):
        self._task.dependent_tasks = tasks

    @property
    def dynamic_tasks(self):
        return self._task.dynamic_tasks

    @dynamic_tasks.setter
    def dynamic_tasks(self, tasks):
        self._task.dynamic_tasks = tasks

    @property
    def future_tasks(self):
        return self._task.future_tasks

    @future_tasks.setter
    def future_tasks(self, tasks):
        self._task.future_tasks = tasks

    @property
    def wcet(self):
        return self._task.wcet

    @property
    def wcbu(self):
        return self._task.wcbu

    @property
    def queue_time(self):
        return self._task.queue_time

    @queue_time.setter
    def queue_time(self, time):
        self._task.queue_time = time

    @property
    def release_time(self):
        return self._task.release_time

    @release_time.setter
    def release_time(self, time):
        self._task.release_time = time

    @property
    def completion_time(self):
        return self._task.completion_time

    @completion_time.setter
    def completion_time(self, time):
        self._task._completion_time = time

    @property
    def execution_time(self):
        return self._task.execution_time

    @execution_time.setter
    def execution_time(self, time):
        self._task.execution_time = time

    def execute(self):
        self._task.execute()

    def has_dependencies(self):
        return self._task.has_dependencies()

    def get_dependencies(self):
        return copy.copy(self._task.dependent_tasks)

    def is_dependency(self, task):
        return self._task.is_dependency(task)

    def add_dependency(self, dependency):
        self._task.add_dependency(dependency)

    def remove_dependency(self, dependency):
        self._task.remove_dependency(dependency)

    def is_dummy(self):
        return self._task.is_dummy()

    def is_sleep_task(self):
        return self._task.is_sleep_task()

    def is_periodic(self):
        return self._task.is_periodic()


class AntTask(TaskDecorator):

    def __init__(self, task: AbstractTask, **kwargs):
        super().__init__(task)
        self.visited_by = []
        self._dependent_tasks = []

    @property
    def dependent_tasks(self):
        return self._dependent_tasks

    def get_dependencies(self):
        return copy.copy(self.dependent_tasks)

    def add_dependency(self, dependency):
        self._dependent_tasks.append(dependency)

    def accept(self, ant, timestamp):
        self.visited_by.append(ant)
        ant.visit(self, timestamp)


class Schedule(list):

    def __init__(self, tasks:List[AbstractTask]=None):
        super().__init__()
        self._dependent_tasks = []
        self._independent_tasks = []
        self._periodic_tasks = []
        self._generated_periodic_tasks = []

        if tasks is not None:
            for task in tasks:
                self.append(task)

    def append(self, task: AbstractTask) -> None:
        if not isinstance(task, AbstractTask):
            raise TypeError("Only Tasks can be added to a schedule")

        # Track Dependencies
        if task.has_dependencies():
            self._dependent_tasks.append(task)
        else:
            self._independent_tasks.append(task)

        # Track Periodic Tasks
        if task.periodicity > timedelta(seconds=0):
            self._periodic_tasks.append(task)
        elif task.periodicity < timedelta(seconds=0):
            self._generated_periodic_tasks.append(task)

        super().append(task)

        if len(self.dependent_tasks) + len(self.independent_tasks) != len(self):
            raise RuntimeError("Schedule Internal Data Invalid. Number of dependent/independent task does not match total tasks")

    def insert(self, __index: int, task: AbstractTask) -> None:
        if not isinstance(task, AbstractTask):
            raise TypeError("Only Tasks can be added to a schedule")

        # Track Dependencies
        if task.has_dependencies():
            self._dependent_tasks.append(task)
        else:
            self._independent_tasks.append(task)

        # Track Periodic Tasks
        if task.periodicity > timedelta(seconds=0):
            self._periodic_tasks.append(task)
        elif task.periodicity < timedelta(seconds=0):
            self._generated_periodic_tasks.append(task)

        super().insert(__index, task)

    def extend(self, schedule: Schedule) -> None:

        if not isinstance(schedule, Schedule):
            raise TypeError("Only Tasks can be added to a schedule")

        self._dependent_tasks.extend(schedule.dependent_tasks)
        self._independent_tasks.extend(schedule.independent_tasks)
        self._periodic_tasks.extend(schedule.periodic_tasks)
        self._generated_periodic_tasks.extend(schedule.generated_periodic_tasks)
        super().extend(schedule)

    @property
    def dependent_tasks(self):
        return self._dependent_tasks

    @property
    def independent_tasks(self):
        return self._independent_tasks

    @property
    def periodic_tasks(self):
        return self._periodic_tasks
    @property
    def generated_periodic_tasks(self):
        return self._generated_periodic_tasks
