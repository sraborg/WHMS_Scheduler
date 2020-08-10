import abc


class AbstractTask:

    def __init__(self):
        self._wcet = None
        self._wcbu = None
        self._deadline = None
        self._nu = None
        self._cost = None
        self._dependent_tasks = None
        self._dynamic_tasks = None          # potential tasks
        self._new_tasks = None              # Actual tasks

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def dependent_tasks(self, cost):
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
    def new_tasks(self):
        return self._new_tasks

    @new_tasks.setter
    def new_tasks(self, tasks):
        self._new_tasks = tasks

    def execute(self):
        pass

