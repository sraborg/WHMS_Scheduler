from dummy_analysis import DummyAnalysis


class AnalysisFactory:

    @staticmethod
    def get_analysis(analysis_type: str):
        analysis = None

        if analysis_type.upper() == "DUMMY":
            analysis = DummyAnalysis()
        else:
            raise Exception("Invalid Analysis Type")

        return analysis
