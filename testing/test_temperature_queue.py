import unittest
from scheduler import TemperatureQueue


class TestAntSchedulers(unittest.TestCase):

    def test_remove_top_1(self):
        """ Verify TemperatureQueue::remove_top actually removes the largest temperature.
            Case: Largest temperature is 35.
            Expected: Pass
        """
        tq = TemperatureQueue()

        tq.push(19)
        tq.push(32)
        tq.push(35)
        tq.push(23)

        largest = tq.pop()
        self.assertEqual(largest, 35)

    def test_remove_top_(self):
        """ Verify TemperatureQueue::remove_top removes the correct number of temperatures.
            Case: Remove 2 items and verify queue removes 2 items
            Expected: Pass
        """
        tq = TemperatureQueue()

        tq.push(19)
        tq.push(32)
        tq.push(35)
        tq.push(23)
        tq.push(82)
        tq.push(49)

        before = len(tq)
        tq.remove_top(2)
        after = len(tq)
        self.assertEqual(before, after + 2)

    def test_peek_(self):
        """ Verify TemperatureQueue::peak returns largest temperature without removing it
            Case: 82 is the largest temperature and 49 is the second largest.
            Expected: Pass
        """
        tq = TemperatureQueue()

        tq.push(19)
        tq.push(32)
        tq.push(35)
        tq.push(23)
        tq.push(82)
        tq.push(49)

        p1 = tq.peek()
        self.assertEqual(p1, 82)
        p2 = tq.peek()
        self.assertEqual(p1, p2)
