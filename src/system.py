from abstract_task import AbstractTask


class System:

    def __init__(self):
        self._scheduler = None
        self._taskset = set()
        self._schedule = []

    def addTask(self, task: AbstractTask):
        self._taskset.add(task)


    def executeSchedule(self):
        self._before()
        for task in self._taskset:
            task._before()
            task._execute()
            task._after()
        self._after()