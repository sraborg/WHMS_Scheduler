from abc import ABC, abstractmethod
from typing import Optional
import copy           # Used in task_builder to fix error with getTask
from analysis import AnalysisFactory
from analysis import AbstractAnalysis                      # Type Hint
from nu import AbstractNu                                  # Type Hint
from nu import NuRegression
from time import sleep


class TaskBuilderInterface(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_task(self):
        pass

    @abstractmethod
    def set_analysis(self, analysis_type: str):
        pass

    @abstractmethod
    def set_soft_deadline(self, deadline: int):
        pass

    @abstractmethod
    def set_hard_deadline(self, deadline: int):
        pass

    @abstractmethod
    def add_dependencies(self, dependencies):
        pass

    # Accessors Methods
    @abstractmethod
    def analysis(self) -> AbstractAnalysis:
        pass

    @abstractmethod
    def soft_deadline(self) -> int:
        pass

    @abstractmethod
    def hard_deadline(self) -> int:
        pass

    @abstractmethod
    def dependent_tasks(self):
        pass

    @abstractmethod
    def dynamic_tasks(self):
        pass


class AbstractTask(ABC):

    def __init__(self, builder: TaskBuilderInterface):
        self._analysis = builder.analysis
        self._nu = builder.nu
        self._soft_deadline = builder.soft_deadline
        self._hard_deadline = builder.hard_deadline
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
    def deadline(self):
        return self._deadline

    @deadline.setter
    def deadline(self, deadline):
        self._deadline = deadline

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

    @abstractmethod
    def value(self):
        pass


class CustomTask(AbstractTask):

    def __init__(self, builder: TaskBuilderInterface):
        super().__init__(builder)

    def value(self):
        return 1


class DummyTask(AbstractTask):

    def __init__(self):
        super().__init__()
        self._runtime = 5

    def run(self):
        sleep(self._runtime)


class ScheduledTask:

    def __init__(self, task: AbstractTask, queue_time):
        self.task = task
        self.queue_time = queue_time
        self.release_time = None
        self.completion_time = None
        self.execution_time = None

    def value(self):
        return self.task.value()

    def execute(self):
        self.task.execute()


class TaskBuilder(TaskBuilderInterface):

    def __init__(self):
        self._analysis: Optional[AbstractAnalysis] = None
        self._nu: Optional[AbstractNu] = None
        self._soft_deadline = None
        self._hard_deadline = None
        self._dependent_tasks = None
        self._dynamic_tasks = None  # potential tasks
        self._new_tasks = None
        self.reset()

    def reset(self):
        self._analysis = None
        self._nu = NuRegression()                       # Defaults to Regression
        self._soft_deadline = None
        self._soft_deadline = None
        self._dependent_tasks = None
        self._dynamic_tasks = None  # potential tasks
        self._new_tasks = None

    def set_analysis(self, analysis_type: str):
        self._analysis = AnalysisFactory.get_analysis(analysis_type)

    def set_nu(self, method: AbstractNu) -> None:
        self._nu = method

    def set_soft_deadline(self, deadline):
        self._deadline = deadline

    def set_hard_deadline(self, deadline):
        self._hard_deadline = deadline

    def add_dependencies(self, dependencies):
        self._dependent_tasks = dependencies

    def add_dynamic_tasks(self, tasks):
        self._dynamic_tasks = tasks

    def get_task(self):

        build_order = copy.copy(self)
        task = CustomTask(build_order)                    # Temp Fix for reference issue

        # Use decorators to add dependencies / dynamic tasks
        if self._dependent_tasks is not None:
            task = TaskWithDependencies(task)

        if self._dynamic_tasks is not None:
            task = TaskWithDynamicTasks(task)

        self.reset()
        return task

    @property
    def analysis(self):
        return self._analysis

    @property
    def nu(self):
        return self._nu

    @property
    def soft_deadline(self) -> int:
        return self._soft_deadline

    @property
    def hard_deadline(self) -> int:
        return self._hard_deadline

    @property
    def dependent_tasks(self):
        return self._dependent_tasks

    @property
    def dynamic_tasks(self):
        return self._dynamic_tasks


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
    def cost(self, method):
        self._task.nu = method

    @property
    def analysis(self):
        return self._task.analysis

    @analysis.setter
    def deadline(self, analysis_type):
        self._task.analysis = analysis_type

    @property
    def deadline(self):
        return self._task.deadline

    @deadline.setter
    def deadline(self, deadline):
        self._task.deadline = deadline

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