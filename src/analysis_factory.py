from abstract_analysis import AbstractAnalysis
from dummy_analysis import DummyAnalysis


class AnalysisFactory:

    @staticmethod
    def get_analysis(type: str):
        analysis = None

        if type.upper() == "DUMMY":
            analysis = DummyAnalysis()
        else:
            raise Exception("Invalid Analysis Type")

        return analysis
