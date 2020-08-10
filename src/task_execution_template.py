from abc import ABC, abstractmethod


class TaskExecutionTemplate(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def execute(self):
        if self.shouldRunBefore():
            self.before()

        self.run()

        if self.shouldRunAfter():
            self.after()

    @abstractmethod
    def before(self):
        pass

    @abstractmethod
    def after(self):
        pass

    def shouldRunBefore(self):
        return False

    def shouldRunAfter(self):
        return False

    def run(self):
        pass