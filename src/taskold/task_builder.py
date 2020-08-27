from taskold.task_builder_interface import TaskBuilderInterface
from taskold.analysis.analysis_factory import AnalysisFactory
from taskold.task_with_dependencies import TaskWithDependencies
from taskold.task_with_dynamic_tasks import TaskWithDynamicTasks
from taskold.custom_task import CustomTask
from typing import Optional
from taskold.analysis.abstract_analysis import AbstractAnalysis                      # Type Hint
from taskold.nu.abstract_nu import AbstractNu                                  # Type Hint
from taskold.nu.nu_regression import NuRegression

import copy           # Used to fix error with getTask


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
