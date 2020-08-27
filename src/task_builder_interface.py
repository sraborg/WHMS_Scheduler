from abc import ABC, abstractmethod
from abstract_analysis import AbstractAnalysis # Typing hints


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
