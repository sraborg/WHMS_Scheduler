from abc import ABC, abstractmethod
from abstract_analysis import AbstractAnalysis


class BuilderInterface(ABC):

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
    def set_deadline(self, deadline):
        pass

    @abstractmethod
    def add_dependencies(self):
        pass
