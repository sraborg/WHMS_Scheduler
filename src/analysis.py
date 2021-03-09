from abc import ABC, abstractmethod
import random
from time import sleep


class AbstractAnalysis(ABC):

    def __init__(self, **kwargs):
        self.wcet = kwargs.get("wcet", 1)
        self.wcbu = kwargs.get("wcbu", 1)

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class DummyAnalysis(AbstractAnalysis):
    """A dummy Analysis Class

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.precision = kwargs.get("precision", (0, 0))

    def execute(self):
        """This function simulates running a dummy taskold.

        :return: void

        The function randoms an execution time close to worst-case execution time (wcet). Note it can exceed it's deadline.

        """
        lower_bound = self.wcet - self.precision[0]
        upper_bound = self.wcet + self.precision[1]

        execution_time = random.uniform(lower_bound, upper_bound)
        print("Running Task: " + str(id(self)))
        sleep(execution_time)
        print("Completed Task " + str(id(self)) + " after " + str(execution_time) + " seconds")

    def name(self):
        return "DUMMY"


class MedicalAnalysis(AbstractAnalysis):

    def name(self) -> str:
        pass

    def execute(self):
        print("Running " + self.name() + " Analysis")
        sleep(self.wcet)


class HeartRateAnalysis(MedicalAnalysis):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wcet = 120
        self.wcbu = 1

    def name(self) -> str:
        return "HEART_RATE"


class BloodPressureAnalysis(MedicalAnalysis):
    def name(self) -> str:
        return "BLOOD_PRESSURE"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wcet = 60
        self.wcbu = 1


class RespiratoryRateAnalysis(MedicalAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wcet = 300
        self.wcbu = 1

    def name(self) -> str:
        return "Respiratory Rate"


class ElectrocardiogramAnalysis(MedicalAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wcet = 10
        self.wcbu = 1

    def name(self) -> str:
        return "Electrocardiogram (ECG)"


class Electroencephalography(MedicalAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wcet = 1800
        self.wcbu = 1

    def name(self) -> str:
        return "Electroencephalography (EEG)"


class BodyTemperatureAnalysis(MedicalAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wcet = 15
        self.wcbu = 1

    def name(self) -> str:
        return "Body Temperature"


class MotionAnalysis(MedicalAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wcet = 600
        self.wcbu = 1

    def name(self) -> str:
        return "Motion"


class BloodGlucoseAnalysis(MedicalAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wcet = 20
        self.wcbu = 1

    def name(self) -> str:
        return "Blood Glucose"


class ExerciseAnalysis(MedicalAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wcet = 1200
        self.wcbu = 1

    def name(self) -> str:
        return "Exercise"


class StressAnalysis(MedicalAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wcet = 480
        self.wcbu = 1

    def name(self) -> str:
        return "Stress"


class SleepAnalysis(AbstractAnalysis):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def execute(self):
        """This function simulates running a dummy taskold.

        :return: void

        The function randoms an execution time close to worst-case execution time (wcet). Note it can exceed it's deadline.

        """
        print("Waiting... ")
        sleep(self.wcet)

    def name(self):
        return "SLEEP"


class AnalysisFactory:

    @staticmethod
    def get_analysis(analysis_type: str):
        analysis = None

        if analysis_type.upper() == "DUMMY":
            analysis = DummyAnalysis()
        elif analysis_type.upper() == "SLEEP":
            analysis = SleepAnalysis()
        elif analysis_type.upper() == "HEART_RATE":
            analysis = HeartRateAnalysis()
        elif analysis_type.upper() == "BLOOD_PRESSURE":
            analysis = BloodPressureAnalysis()
        else:
            raise Exception("Invalid Analysis Type")

        return analysis

    @classmethod
    def dummy_analysis(cls):
        return DummyAnalysis()

    @classmethod
    def sleep_analysis(cls):
        return SleepAnalysis()

    @classmethod
    def heart_rate_analysis(cls):
        return HeartRateAnalysis()

    @classmethod
    def blood_pressure_analysis(cls):
        return BloodPressureAnalysis()

    @classmethod
    def respiratory_rate_analysis(cls):
        return RespiratoryRateAnalysis()

    @classmethod
    def electrocardiogram_analysis(cls):
        return ElectrocardiogramAnalysis()

    @classmethod
    def electroencephalography_analysis(cls):
        return Electroencephalography()

    @classmethod
    def body_temperature_analysis(cls):
        return BodyTemperatureAnalysis()

    @classmethod
    def motion_analysis(cls):
        return MotionAnalysis()

    @classmethod
    def blood_glucose_analysis(cls):
        return BloodGlucoseAnalysis()

    @classmethod
    def exercise_analysis(cls):
        return ExerciseAnalysis()

    @classmethod
    def stress_analysis(cls):
        return StressAnalysis()