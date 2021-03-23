import unittest
from src.scheduler import *
from src.task import SleepTask
from src.nu import NuFactory
from src.analysis import MedicalAnalysis


class Nu(unittest.TestCase):

    def test_shift_deadlines(self):
        start_time = datetime.now()
        tasks = Schedule()
        t1 = UserTask()
        t1.values = [
            (start_time.timestamp(), 1),
            ((start_time + timedelta(seconds=30)).timestamp(), 1),
            ((start_time + timedelta(seconds=60)).timestamp(), 1)
        ]
        t2 = copy.deepcopy(t1)

        nu = NuFactory.get_nu("CONSTANT", CONSTANT_VALUE=1)
        pass