import unittest
from src.scheduler import *
from src.task import SleepTask
from src.nu import NuFactory
from src.analysis import MedicalAnalysis
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly


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

    def test_fit_model(self):
        earliest_start = datetime.now()
        soft_deadline = (earliest_start + timedelta(seconds=30))
        hard_deadline = (earliest_start + timedelta(seconds=60))
        t = UserTask()
        t.values = [
            (earliest_start.timestamp(), 1),
            (soft_deadline.timestamp(), 100),
            (hard_deadline.timestamp(), 1)
        ]

        t.nu = NuFactory.regression()
        t.nu.fit_model(t.values)
        t.value(earliest_start)
        t.value(soft_deadline)
        t.value(hard_deadline)

        x_axis_scale = np.linspace(t.values[0], t.values[-1], num=len(t.values) * 10)
        xs = t.nu._x

        #fig1 = plt.figure()
        #ax1 = fig1.add_subplot(111)
        #ax1.scatter(x, y, facecolors='None')


    def test_numpy_poly(self):
        beg = datetime.now()
        mid = beg + timedelta(minutes=5)
        end = beg + timedelta(minutes=10)
        xs = [beg.timestamp(), mid.timestamp(), end.timestamp()]
        ys = [0, 100, 0]

        coef = poly.polyfit(xs, ys, 2)
        fit = poly.polyval(mid.timestamp(), coef)


        x_line = np.linspace(xs[0], xs[2], 200)
        ffit = poly.polyval(x_line, coef)
        plt.plot(x_line, ffit)
        plt.show()
        print("t")
