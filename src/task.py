from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Tuple
import copy           # Used in task_builder to fix error with getTask
from nu import NuFactory, NuRegression
from analysis import AnalysisFactory
from analysis import AbstractAnalysis                      # Type Hint
from nu import AbstractNu                                  # Type Hint
from nu import NuRegression
from time import sleep


class AbstractTaskBuilder(ABC):

    def __init__(self):
        self._analysis: Optional[AbstractAnalysis] = None
        self._nu: Optional[AbstractNu] = None
        self._earliest_start: datetime = None
        self._soft_deadline: datetime = None
        self._hard_deadline: datetime = None
        self._dependent_tasks = None
        self._dynamic_tasks = None  # potential tasks
        self._new_tasks = None
        self.reset()

    def reset(self):
        self._analysis = None
        self._nu = NuRegression()                       # Defaults to Regression
        self._earliest_start = None
        self._soft_deadline = None
        self._hard_deadline = None
        self._dependent_tasks = None
        self._dynamic_tasks = None  # potential tasks
        self._new_tasks = None

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

    def __init__(self, builder: AbstractTaskBuilder):
        self._analysis = builder.analysis
        self._nu = builder.nu
        self._earliest_start: datetime = builder.earliest_start
        self._soft_deadline: datetime = builder.soft_deadline
        self._hard_deadline: datetime = builder.hard_deadline
        self._cost = None
        self._dependent_tasks = builder.dependent_tasks
        self._dynamic_tasks = builder.dynamic_tasks        # potential tasks
        self._future_tasks = []                           # Tasks

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, cost):
        self._cost = cost

    @property
    def analysis(self):
        return self._analysis

    @analysis.setter
    def analysis(self, analysis):
        self._analysis = analysis

    @property
    def earliest_start(self) -> datetime:
        return self._earliest_start

    @earliest_start.setter
    def earliest_start(self, deadline: datetime):
        self._earliest_start = deadline
    @property
    def soft_deadline(self) -> datetime:
        return self._soft_deadline

    @soft_deadline.setter
    def soft_deadline(self, deadline: datetime):
        self._soft_deadline = deadline

    @property
    def hard_deadline(self) -> datetime:
        return self._hard_deadline

    @hard_deadline.setter
    def hard_deadline(self, deadline: datetime):
        self._hard_deadline = deadline

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

    def execute(self):
        self._analysis.execute()

    def value(self):
        return self._nu.eval(datetime.now().timestamp())


class CustomTask(AbstractTask):

    def __init__(self, builder: AbstractTaskBuilder):
        super().__init__(builder)


class DummyTask(AbstractTask):

    def __init__(self):
        super().__init__()
        self._runtime = 5

    def execute(self):
        sleep(self._runtime)

    def value(self):
        return 0


class ScheduledTask:

    def __init__(self, task: AbstractTask, queue_time):
        self.task = task
        self.queue_time: datetime = queue_time
        self.release_time: datetime = None
        self.completion_time: datetime = None
        self.execution_time: datetime = None

    def value(self):
        return self.task.value()

    def execute(self):
        self.task.execute()


class TaskBuilder(AbstractTaskBuilder):

    def __init__(self):
        super().__init__()

    def build_task(self):

        build_order = copy.copy(self)
        task = CustomTask(build_order)                    # Temp Fix for reference issue

        # Use decorators to add dependencies / dynamic tasks
        if self._dependent_tasks is not None:
            task = TaskWithDependencies(task)

        if self._dynamic_tasks is not None:
            task = TaskWithDynamicTasks(task)

        self.reset()
        return task


class TaskDecorator(ABC):

    def __init__(self, task: AbstractTask):
        super().__init__()
        self._task = task

    def value(self):
        return self._task.value()

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

    def execute(self):
        self._task.execute()


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