from abstract_task import AbstractTask


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