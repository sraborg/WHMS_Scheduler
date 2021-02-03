from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import pickle
import csv
import json
import copy           # Used in task_builder to fix error with getTask
from nu import NuFactory, NuRegression, NuConstant
from analysis import AnalysisFactory
from analysis import AbstractAnalysis, DummyAnalysis, SleepAnalysis
from nu import AbstractNu                                  # Type Hint
from nu import NuRegression
#from scheduler import Ant
import random
from time import sleep


class AbstractTaskBuilder(ABC):

    def __init__(self):
        self._analysis: Optional[AbstractAnalysis] = None
        self._nu: Optional[AbstractNu] = None
        self._earliest_start: datetime = None
        self._soft_deadline: datetime = None
        self._hard_deadline: datetime = None
        self._dependent_tasks = []
        self._dynamic_tasks = None  # potential tasks
        self._new_tasks = None
        self._periodic = False
        self.reset()

    def reset(self):
        self._analysis = None
        self._nu = NuRegression()                       # Defaults to Regression
        self._earliest_start = None
        self._soft_deadline = None
        self._hard_deadline = None
        self._dependent_tasks = []
        self._dynamic_tasks = None  # potential tasks
        self._new_tasks = None
        self._periodic = False

    @abstractmethod
    def build_task(self):
        pass

    def set_analysis(self, analysis_type: str):
        self._analysis = AnalysisFactory.get_analysis(analysis_type)

    def set_nu(self, method: str) -> None:
        self._nu = NuFactory.get_nu("Regression")

    def fit_model(self, values: List[Tuple[datetime, int]]):
        self._nu.fit_model(values)

    def set_earliest_start(self, start_time: datetime):
        self._earliest_start = start_time

    def set_soft_deadline(self, time):
        self._soft_deadline = time

    def set_hard_deadline(self, time):
        self._hard_deadline = time

    def set_periodic(self):
        self._periodic = True

    def add_dependencies(self, dependencies):
        self._dependent_tasks = dependencies

    def add_dynamic_tasks(self, tasks):
        self._dynamic_tasks = tasks

    # Accessors Methods
    @property
    def analysis(self):
        return self._analysis

    @property
    def nu(self):
        return self._nu

    @property
    def earliest_start(self) -> datetime:
        return self._earliest_start

    @property
    def soft_deadline(self) -> datetime:
        return self._soft_deadline

    @property
    def hard_deadline(self) -> datetime:
        return self._hard_deadline

    @property
    def dependent_tasks(self):
        return self._dependent_tasks

    @property
    def dynamic_tasks(self):
        return self._dynamic_tasks


class AbstractTask(ABC):

    """
    def __init__(self, builder: AbstractTaskBuilder):
        self.analysis = builder.analysis
        self.ordered_by = ""
        self.nu = builder.nu
        self.earliest_start: datetime = builder.earliest_start
        self.soft_deadline: datetime = builder.soft_deadline
        self.hard_deadline: datetime = builder.hard_deadline
        self.periodicity: 0
        self._cost = None
        self._dependent_tasks: List[AbstractTask] = builder.dependent_tasks
        self._dynamic_tasks = builder.dynamic_tasks        # potential tasks
        self._future_tasks = []                           # Tasks
        self._queue_time = None
        self._release_time: datetime = None
        self._completion_time: datetime = None
        self._execution_time: datetime = None
    """

    def __init__(self, **kwargs):
        self.analysis = kwargs.get("analysis", None)
        self.ordered_by = kwargs.get("ordered_by", "")
        self.values = kwargs.get("values", [])
        self.nu = kwargs.get("nu", None)
        self.earliest_start: datetime = kwargs.get("earliest_start", None)
        self.soft_deadline: datetime = kwargs.get("soft_deadline", None)
        self.hard_deadline: datetime = kwargs.get("hard_deadline", None)
        self.periodicity: int = kwargs.get("periodicity", 0)
        self._cost = kwargs.get("cost", None)
        self._dependent_tasks: List[AbstractTask] = kwargs.get("dependent_tasks", [])
        self._dynamic_tasks = kwargs.get("dynamic_tasks", [])        # potential tasks
        self._future_tasks = []                           # Tasks
        self._queue_time = None
        self._release_time: datetime = None
        self._completion_time: datetime = None
        self._execution_time: datetime = None

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

    def execute(self):
        self.analysis.execute()

    def value(self, **kwargs):
        if "timestamp" in kwargs:
            timestamp = kwargs.get("timestamp")
        else:
            timestamp = datetime.now().timestamp()
        return self.nu.eval(timestamp)

    def has_dependencies(self):
        return not not self._dependent_tasks

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

    """
    @staticmethod
    def load_tasks(path):
        tasks = []
        tb = TaskBuilder()
        with open(path) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=",")

            for row in csv_reader:

                task = UserTask()

                # Setup Analysis
                task.analysis = AnalysisFactory.get_analysis(row["analysis"])

                task.earliest_start = datetime.fromtimestamp(float(row["earliest_start"]))
                task.soft_deadline = datetime.fromtimestamp(float(row["soft_deadline"]))
                task.hard_deadline = datetime.fromtimestamp(float(row["hard_deadline"]))
                task.ordered_by = row["ordered_by"]

                # Setup Nu


                # Handle Dependencies
                dependent_task_indices = row["dependent_tasks"].split(";")
                for dependency in dependent_task_indices:
                    if len(dependency) > 0:
                        index = int(dependency)
                        task.add_dependency(tasks[index])

                tasks.append(task)
        return tasks
    """

    @staticmethod
    def load_tasks(path):
        tasks = []
        tb = TaskBuilder()
        with open(path, 'r') as json_file:
            loaded_tasks = json.load(json_file)

            for entry in loaded_tasks:

                task = UserTask()

                # Setup Analysis
                task.analysis = AnalysisFactory.get_analysis(entry["analysis"])

                task.earliest_start = datetime.fromtimestamp(float(entry["earliest_start"]))
                task.soft_deadline = datetime.fromtimestamp(float(entry["soft_deadline"]))
                task.hard_deadline = datetime.fromtimestamp(float(entry["hard_deadline"]))
                task.ordered_by = entry["ordered_by"]
                task.values = entry["values"]

                # Setup Nu
                nu = NuFactory.get_nu(entry["nu"])
                nu.fit_model(task.values)
                task.nu = nu

                # Handle Dependencies
                dependent_task_indices = entry["dependent_tasks"].split(";")
                for dependency in dependent_task_indices:
                    if len(dependency) > 0:
                        index = int(dependency)
                        task.add_dependency(tasks[index])

                tasks.append(task)
        return tasks



    """
    @staticmethod
    def save_tasks(path, tasklist):
        with open(path, mode='w') as csv_file:
            fieldnames = [
                "analysis",
                "earliest_start",
                "earliest_start_value",
                "soft_deadline",
                "soft_deadline_value",
                "hard_deadline",
                "hard_deadline_value",
                "ordered_by",
                "dependent_tasks"
            ]
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

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
                    #'earliest_start_value': task.nu.
                    #'soft_deadline': task.soft_deadline.timestamp(),
                    #'hard_deadline': task.hard_deadline.timestamp(),
                    'ordered_by': task.ordered_by,
                    'dependent_tasks': dependent,
                    }

                # Determine Values
                if isinstance(task.nu, NuRegression):
                    pass
                elif isinstance(task.nu, NuConstant):
                    entry['value'] = task.nu.eval()

                csv_writer.writerow(entry)
    """

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
            nu = NuFactory.regression()
            nu.fit_model(t.values)
            t.nu = nu

            analysis_types = [
                "BLOOD_PRESSURE",
                "HEART_RATE"
            ]
            t.analysis = AnalysisFactory.get_analysis(random.choice(analysis_types))

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

            if dependencies and i > 0 and random.randint(0, 1) == 1:
                t.add_dependency(tasklist[i - 1])
            tasklist.append(t)

        return tasklist


class UserTask(AbstractTask):

    def __init__(self, builder: AbstractTaskBuilder=None, **kwargs):
        """
        if builder is None:
            builder = DummyTaskBuilder()

        super().__init__(builder)
        """
        super().__init__(**kwargs)

    def name(self):
        return "User"


class SystemTask(AbstractTask):
    pass


class DummyTask(AbstractTask):

    def __init__(self, builder: AbstractTaskBuilder = None, **kwargs):

        if builder is None:
            builder = DummyTaskBuilder()

        super().__init__(builder)
        if "runtime" in kwargs:
            self.runtime = kwargs.get("runtime")
        else:
            self.runtime = 5

        if "analysis_type" in kwargs and str.upper(kwargs.get("analysis_type")) == "SLEEPANALYSIS":
            self.analysis = SleepAnalysis()
        else:
            self.analysis = DummyAnalysis(wcet=self.runtime)

    def execute(self):
        self.analysis.execute()
        #sleep(self.runtime)

    def name(self):
        return "DUMMY"


class SleepTask(SystemTask):

    def __init__(self, builder: AbstractTaskBuilder = None, **kwargs):

        if builder is None:
            builder = DummyTaskBuilder()

        super().__init__()
        if "wcet" in kwargs:
            wcet = kwargs.get("wcet")
        else:
            wcet = 5

        if "analysis_type" in kwargs and str.upper(kwargs.get("analysis_type")) == "SLEEPANALYSIS":
            self.analysis = SleepAnalysis(wcet=wcet)
        else:
            self.analysis = DummyAnalysis(wcet=wcet)

        # Set all points to now
        self.earliest_start = datetime.now()
        self.soft_deadline = self.earliest_start
        self.hard_deadline = self.earliest_start

    def execute(self):
        self.analysis.execute()
        #sleep(self.runtime)

    def value(self, **kwargs):
        return 0

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


class TaskWithPeriodicity(TaskDecorator):

    def __init__(self, task: AbstractTask, **kwargs):
        super().__init__(task)
        self._peridic_interval = None
        if "periodic_interval" in kwargs:
            self._peridic_interval = kwargs.get("periodic_interval")
        else:
            raise ValueError("No Periodic Interval Set")

    def is_periodic(self):
        return True


class TaskWithDependencies(TaskDecorator):

    def __init__(self, task: AbstractTask):
        super().__init__(task)

    def execute(self):
        # something with dependenciesa
        self._task.execute()


class TaskWithDynamicTasks(TaskDecorator):

    def __init__(self, task: AbstractTask):
        super().__init__(task)

    def execute(self):
        self._task.execute()
        self._check_dynamic_tasks()

    def _check_dynamic_tasks(self):
        for dyn_task in self.dynamic_tasks:
            if dyn_task[1]:
                tasks = self._task.future_tasks.append(dyn_task[0])


class TaskBuilder(AbstractTaskBuilder):

    def __init__(self):
        super().__init__()

    def build_task(self):

        build_order = copy.copy(self)
        task = UserTask(build_order)                    # Temp Fix for reference issue

        if self.set_periodic():
            task = TaskWithPeriodicity(task)

        # Use decorators to add dependencies / dynamic tasks
        if self._dependent_tasks is not None:
            task = TaskWithDependencies(task)

        if self._dynamic_tasks is not None:
            task = TaskWithDynamicTasks(task)

        self.reset()
        return task


''' Builder for Generating DummyTasks 

'''


class DummyTaskBuilder(AbstractTaskBuilder):

    def __init__(self):
        super().__init__()

    def build_task(self):
        task = Task(self)  # Temp Fix for reference issue
        return task


class TaskFactory():

    @staticmethod
    def get_task(task_type: str, **kwargs) -> AbstractTask:

        if task_type.upper() == "SLEEP":
            return SleepTask(**kwargs)
        elif task_type.upper() == "DUMMY":
            return DummyTask()
        else:
            return Task()

    @classmethod
    def sleep_task(cls, wcet=5):
        task = Task()
        task.analysis = SleepAnalysis(wcet=wcet)
        return SleepTask()