import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from selector import select_threshold, route


class SelectorTest(unittest.TestCase):
    def test_zero_budget_keeps_ties(self):
        tau, lost = select_threshold([0, 1, 2, 3], [1, 2, 0],
                                     [True, True, False], [True, True, True], 0)
        self.assertEqual((tau, lost), (1, 0))

    def test_denominator_is_all_standard(self):
        tau, lost = select_threshold([0, 1, 2, 3], [1, 2, 0, 0],
                                     [True]*4, [True, True, False, False], .25)
        self.assertEqual((tau, lost), (2, 1))

    def test_infinite_endpoints(self):
        self.assertEqual(select_threshold([1,2], [0], [True], [True], 0), (-np.inf, 0))
        self.assertEqual(select_threshold([1,2], [0], [True], [True], 1), (np.inf, 1))

    def test_routing_without_gold_preserves_response(self):
        rows = [dict(sid='a', nota='E'), dict(sid='b', nota='C')]
        answers = [dict(sid='a', response='D. original answer'), dict(sid='b', response='B')]
        result = route(rows, answers, [1, 0], 1)
        self.assertEqual(result[0]['response'], 'D. original answer')
        self.assertEqual(result[1]['response'], 'C')

    def test_misaligned_responses_rejected(self):
        with self.assertRaises(ValueError):
            route([dict(sid='a', nota='E')], [dict(sid='b', response='A')], [0], 1)


if __name__ == '__main__':
    unittest.main()
