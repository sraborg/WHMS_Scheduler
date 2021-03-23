import unittest
from scheduler import MetaHeuristicScheduler


class TestMetaheuristicSchedulers(unittest.TestCase):

    def test_permutate_schedule_by_blockinsert_preserves_list_size(self):
        """
        Tests if the Metaheuristic Scheduler method "permute_by_block_insert" preserves the list size.
        :return: Returns True if the size remains the same, otherwise false
        """
        l = [1,2,3,4,5,6,7,8,9,0]
        l2 = l[:]
        scheduler = MetaHeuristicScheduler()
        MetaHeuristicScheduler.permute_schedule_by_blockinsert(l)

        self.assertEqual(len(l), len(l2))

    def test_permutate_schedule_by_reverse_preserves_list_size(self):
        """
        Tests if the Metaheuristic Scheduler method "permute_by_block_reverse" preserves the list size.
        :return: Returns True if the size remains the same, otherwise false
        """
        l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        l2 = l[:]
        scheduler = MetaHeuristicScheduler()
        MetaHeuristicScheduler.permute_schedule_by_inverse(l)

        self.assertEqual(len(l), len(l2))

    def test_permutate_schedule_by_transport_preserves_list_size(self):
        """
        Tests if the Metaheuristic Scheduler method "permute_by_block_reverse" preserves the list size.
        :return: Returns True if the size remains the same, otherwise false
        """
        l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        l2 = l[:]
        scheduler = MetaHeuristicScheduler()
        MetaHeuristicScheduler.permute_schedule_by_transport(l)

        self.assertEqual(len(l), len(l2))
