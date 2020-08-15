from builder_interface import BuilderInterface
from analysis_factory import AnalysisFactory
from abstract_analysis import AbstractAnalysis
from task_with_dependencies import TaskWithDependencies
from task_with_dynamic_tasks import TaskWithDynamicTasks
from task import Task


class TaskBuilder(BuilderInterface):

    def __init__(self):
        self._analysis = None
        self._deadline = None
        self._dependent_tasks = None
        self._dynamic_tasks = None  # potential tasks
        self._new_tasks = None
        self.reset()

    def reset(self):
        self._analysis = None
        self._deadline = None
        self._dependent_tasks = None
        self._dynamic_tasks = None  # potential tasks
        self._new_tasks = None

    def set_analysis(self, analysis_type: str):
        self._analysis = AnalysisFactory.get_analysis(analysis_type)

    def set_deadline(self, deadline):
        self._deadline = deadline

    def add_dependencies(self, dependencies):
        self._dependent_tasks = dependencies

    def add_dynamic_tasks(self, tasks):
        self._dynamic_tasks = tasks

    def get_task(self):

        task = Task(self)


        # Use decorators to add dependencies / dynamic tasks
        if self._dependent_tasks is not None:
            task = TaskWithDependencies(task)

        if self._dynamic_tasks is not None:
            task = TaskWithDynamicTasks(task)

        self.reset()
        return task
