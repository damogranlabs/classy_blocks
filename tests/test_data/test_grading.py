import unittest
from parameterized import parameterized

import os

from classy_blocks.util import grading_calculator as gc
from classy_blocks import grading
from classy_blocks.grading import Grading

# numbers are calculated with the calculator all this is 'borrowed' from
# https://openfoamwiki.net/index.php/Scripts/blockMesh_grading_calculation
# with a few differences:
# - scipy.optimize.<whatever> can be used here instead of barbarian bisection
# - all floats are converted to integers by rounding down (only matters for border cases)
class TestGradingCalculator(unittest.TestCase):
    def test_get_start_size__count__c2c_expansion(self):
        # valid cases
        self.assertEqual(gc.get_start_size__count__c2c_expansion(1, 10, 1), 0.1)
        self.assertAlmostEqual(gc.get_start_size__count__c2c_expansion(1, 10, 1.1), 0.06274539488, places=5)

        # invalid cases
        with self.assertRaises(AssertionError):
            gc.get_start_size__count__c2c_expansion(0, 10, 1)

        with self.assertRaises(AssertionError):
            gc.get_start_size__count__c2c_expansion(1, 0.5, 1)

    def test_get_start_size__end_size__total_expansion(self):
        self.assertEqual(gc.get_start_size__end_size__total_expansion(1, 0.1, 1), 0.1)

        with self.assertRaises(AssertionError):
            gc.get_start_size__end_size__total_expansion(0, 0.1, 1)

        with self.assertRaises(AssertionError):
            gc.get_start_size__end_size__total_expansion(1, 0.1, 0)

    def test_get_end_size__start_size__total_expansion(self):
        self.assertEqual(gc.get_end_size__start_size__total_expansion(1, 0.1, 10), 1)

        with self.assertRaises(AssertionError):
            gc.get_end_size__start_size__total_expansion(-1, 0.1, 0)

    def test_get_count__start__size__c2c_expansion(self):
        # valid cases
        self.assertEqual(gc.get_count__start_size__c2c_expansion(1, 1, 1), 2)
        self.assertEqual(gc.get_count__start_size__c2c_expansion(1, 0.1, 1), 11)
        self.assertEqual(gc.get_count__start_size__c2c_expansion(1, 0.1, 1.1), 8)

        # border cases
        self.assertEqual(gc.get_count__start_size__c2c_expansion(1, 2, 1), 1)
        self.assertEqual(gc.get_count__start_size__c2c_expansion(1, 1, 2), 2)

        # invalid cases
        with self.assertRaises(AssertionError):
            gc.get_count__start_size__c2c_expansion(0, 0.1, 1.1)  # length < 0

        with self.assertRaises(AssertionError):
            gc.get_count__start_size__c2c_expansion(1, 0, 1.1)  # start_size = 0

        with self.assertRaises(ValueError):
            gc.get_count__start_size__c2c_expansion(1, 0.95, 0)  # c2c_expansion < 1

    def test_get_count__end_size__c2c_expansion(self):
        # valid cases
        self.assertEqual(gc.get_count__end_size__c2c_expansion(1, 0.1, 1), 11)
        self.assertEqual(gc.get_count__end_size__c2c_expansion(1, 0.1, 1.1), 26)
        self.assertEqual(gc.get_count__end_size__c2c_expansion(1, 0.1, 0.9), 8)

        # border cases
        self.assertEqual(gc.get_count__end_size__c2c_expansion(1, 1, 1), 2)
        self.assertEqual(gc.get_count__end_size__c2c_expansion(1, 1, 2), 2)

        # invalid cases
        with self.assertRaises(ValueError):
            gc.get_count__end_size__c2c_expansion(1, 0.1, 1.5)

    def test_get_count__total_expansion__c2c_expansion(self):
        # valid cases
        self.assertEqual(gc.get_count__total_expansion__c2c_expansion(1, 3, 1.1), 12)

        # border cases
        self.assertEqual(gc.get_count__total_expansion__c2c_expansion(1, 1, 1.1), 1)

        # invalid cases
        with self.assertRaises(AssertionError):
            gc.get_count__total_expansion__c2c_expansion(1, 1, 1)

        with self.assertRaises(AssertionError):
            gc.get_count__total_expansion__c2c_expansion(1, -1, 1.1)

    def test_get_count__total_expansion__start_size(self):
        # valid cases
        self.assertEqual(gc.get_count__total_expansion__start_size(1, 1, 0.1), 10)
        self.assertEqual(gc.get_count__total_expansion__start_size(1, 2, 0.1), 7)
        self.assertEqual(gc.get_count__total_expansion__start_size(1, 8, 0.1), 3)

        # border cases
        self.assertEqual(gc.get_count__total_expansion__start_size(1, 0.9, 0.5), 3)
        self.assertEqual(gc.get_count__total_expansion__start_size(1, 0.3, 1), 2)

    def test_get_c2c_expansion__count__start_size(self):
        # valid cases
        self.assertEqual(gc.get_c2c_expansion__count__start_size(1, 10, 0.1), 1)
        self.assertEqual(gc.get_c2c_expansion__count__start_size(1, 2, 0.1), 9)
        self.assertAlmostEqual(gc.get_c2c_expansion__count__start_size(1, 5, 0.1), 1.352395572, places=5)
        self.assertEqual(gc.get_c2c_expansion__count__start_size(1, 2, 0.5), 1)
        self.assertAlmostEqual(gc.get_c2c_expansion__count__start_size(1, 10, 0.05), 1.1469127, places=5)

        # border cases
        self.assertEqual(gc.get_c2c_expansion__count__start_size(1, 1, 0.1), 1)
        self.assertAlmostEqual(gc.get_c2c_expansion__count__start_size(1, 20, 0.1), 0.9181099911, places=5)

        # invalid cases
        with self.assertRaises(AssertionError):
            gc.get_c2c_expansion__count__start_size(0, 1, 0.1)  # length = 0

        with self.assertRaises(AssertionError):
            gc.get_c2c_expansion__count__start_size(1, 0, 0.1)  # count < 1

        with self.assertRaises(AssertionError):
            gc.get_c2c_expansion__count__start_size(1, 10, 1.1)  # start_size > length

        with self.assertRaises(AssertionError):
            gc.get_c2c_expansion__count__start_size(1, 10, 0)  # start_size = 0

        with self.assertRaises(ValueError):
            gc.get_c2c_expansion__count__start_size(1, 10, 0.9)

    def test_get_c2c_expansion__count__end_size(self):
        # valid cases
        self.assertEqual(gc.get_c2c_expansion__count__end_size(1, 10, 0.1), 1)
        self.assertAlmostEqual(gc.get_c2c_expansion__count__end_size(1, 10, 0.01), 0.6784573173, places=5)
        self.assertAlmostEqual(gc.get_c2c_expansion__count__end_size(1, 10, 0.2), 1.202420088, places=5)

        # border cases
        self.assertEqual(gc.get_c2c_expansion__count__end_size(1, 1, 1), 1)

        # invalid cases
        with self.assertRaises(AssertionError):
            gc.get_c2c_expansion__count__end_size(1, 0.5, 1)

        with self.assertRaises(AssertionError):
            gc.get_c2c_expansion__count__end_size(1, 10, -0.5)

        with self.assertRaises(ValueError):
            gc.get_c2c_expansion__count__end_size(1, 10, 1)

    def test_get_c2c_expansion__count__total_expansion(self):
        # valid cases
        self.assertAlmostEqual(gc.get_c2c_expansion__count__total_expansion(1, 10, 5), 1.195813175, places=5)
        self.assertAlmostEqual(gc.get_c2c_expansion__count__total_expansion(1, 10, 0.5), 0.9258747123, places=5)

        # border cases
        self.assertEqual(gc.get_c2c_expansion__count__total_expansion(1, 10, 1), 1)

        # invalid cases
        with self.assertRaises(AssertionError):
            gc.get_c2c_expansion__count__total_expansion(1, 1, 1)

    def test_get_total_expansion__count__c2c_expansion(self):
        # valid cases
        self.assertEqual(gc.get_total_expansion__count__c2c_expansion(1, 10, 1), 1)
        self.assertEqual(gc.get_total_expansion__count__c2c_expansion(1, 1, 1), 1)
        self.assertAlmostEqual(gc.get_total_expansion__count__c2c_expansion(1, 10, 1.1), 2.3579476, places=5)

        # border cases
        self.assertEqual(gc.get_total_expansion__count__c2c_expansion(1, 1, 1), 1)

        # invalid cases
        with self.assertRaises(AssertionError):
            gc.get_total_expansion__count__c2c_expansion(1, 0.5, 1)

    def test_get_total_expansion__start_size__end_size(self):
        self.assertEqual(gc.get_total_expansion__start_size__end_size(1, 1, 1), 1)
        self.assertAlmostEqual(gc.get_total_expansion__start_size__end_size(1, 0.1, 0.01), 0.1)
        self.assertAlmostEqual(gc.get_total_expansion__start_size__end_size(1, 0.01, 0.1), 10)

        with self.assertRaises(AssertionError):
            gc.get_total_expansion__start_size__end_size(1, 0, 0.1)


class TestGrading(unittest.TestCase):
    def setUp(self):
        self.g = Grading(1)

    def test_calculator_functions(self):
        expected_functions = [
            # return_value | param1 | param2 (param0 = length)
            ["c2c_expansion", ["count", "end_size"], gc.get_c2c_expansion__count__end_size],
            ["c2c_expansion", ["count", "start_size"], gc.get_c2c_expansion__count__start_size],
            ["c2c_expansion", ["count", "total_expansion"], gc.get_c2c_expansion__count__total_expansion],
            ["count", ["end_size", "c2c_expansion"], gc.get_count__end_size__c2c_expansion],
            ["count", ["start_size", "c2c_expansion"], gc.get_count__start_size__c2c_expansion],
            ["count", ["total_expansion", "c2c_expansion"], gc.get_count__total_expansion__c2c_expansion],
            ["count", ["total_expansion", "start_size"], gc.get_count__total_expansion__start_size],
            ["end_size", ["start_size", "total_expansion"], gc.get_end_size__start_size__total_expansion],
            ["start_size", ["count", "c2c_expansion"], gc.get_start_size__count__c2c_expansion],
            ["start_size", ["end_size", "total_expansion"], gc.get_start_size__end_size__total_expansion],
            ["total_expansion", ["count", "c2c_expansion"], gc.get_total_expansion__count__c2c_expansion],
            ["total_expansion", ["start_size", "end_size"], gc.get_total_expansion__start_size__end_size],
        ]

        self.assertListEqual(expected_functions, grading.functions)

    @parameterized.expand((
        # [{keys}, count, total_expansion]; length=1 for all cases
        [{"count": 10, "total_expansion": 5}, 10, 5],
        [{"count": 10, "c2c_expansion": 1.1}, 10, 2.357947691],
        [{"count": 10, "c2c_expansion": 0.9}, 10, 0.387420489],
        [{"count": 10, "start_size": 0.2}, 10, 0.1903283012],
        [{"count": 10, "end_size": 0.2}, 10, 5.254123465509412],
        [{"count": 10, "end_size": 0.05}, 10, 0.2912203517],
        [{"total_expansion": 5, "c2c_expansion": 1.1}, 17, 5],
        [{"total_expansion": 0.2, "c2c_expansion": 0.9}, 16, 0.2],
        [{"total_expansion": 0.2, "start_size": 0.1}, 20, 0.2],
        [{"total_expansion": 5, "start_size": 0.1}, 4, 5],
        [{"total_expansion": 5, "end_size": 0.5}, 4, 5],
        [{"total_expansion": 0.2, "end_size": 0.1}, 4, 0.2],
        [{"c2c_expansion": 1.1, "start_size": 0.1}, 8, 1.9487171],
        [{"c2c_expansion": 0.95, "start_size": 0.1}, 14, 0.5133420832],
        [{"c2c_expansion": 1.1, "end_size": 0.1}, 26, 10.8347059433],
        [{"c2c_expansion": 0.95, "end_size": 0.1}, 9, 0.66342043128],
        [{"start_size": 0.1, "end_size": 0.05}, 14, 0.5],
        [{"start_size": 0.05, "end_size": 0.1}, 14, 2],
    ))
    def test_calculate(self, keys, count, total_expansion):
        results = grading.calculate(1, keys)
        self.assertEqual(results[0], count)
        self.assertAlmostEqual(results[1], total_expansion, places=5)

    def add_division(self, length_ratio, count_ratio, total_expansion):
        self.g.divisions.append([length_ratio, count_ratio, total_expansion])

    def test_output_empty(self):
        with self.assertRaises(ValueError):
            str(self.g.grading)

    def test_output_single(self):
        self.add_division(1, 1, 3)
        self.assertEqual(str(self.g.grading), "3")

    def test_output_multi(self):
        self.add_division(0.25, 0.4, 2)
        self.add_division(0.5, 0.2, 1)
        self.add_division(0.25, 0.4, 0.5)

        expected_output = (
            "("
            + os.linesep
            + "\t(0.25 0.4 2)"
            + os.linesep
            + "\t(0.5 0.2 1)"
            + os.linesep
            + "\t(0.25 0.4 0.5)"
            + os.linesep
            + ")"
        )

        self.assertEqual(str(self.g.grading), expected_output)

    def test_copy(self):
        """check that copy doesn't spoil the original"""
        self.add_division(1, 1, 3)
        h = self.g.copy()

        h.divisions[0][2] = 5

        self.assertEqual(self.g.divisions[0][2], 3)

    def test_copy_invert_simple(self):
        self.add_division(1, 1, 5)
        h = self.g.copy(invert=True)

        self.assertEqual(self.g.divisions[0][2], 5)
        self.assertEqual(h.divisions[0][2], 0.2)

    def test_add_division_fail(self):
        with self.assertRaises(AssertionError):
            self.g.length = 0
            self.g.add_division(count=10)

        self.g.length = 1
        with self.assertRaises(ValueError):
            # when using only 1 parameter, c2c_expansion is assumed 1;
            # when specifying that as well, another parameter must be provided
            self.g.add_division(c2c_expansion=1.1)

        with self.assertRaises(AssertionError):
            # specified total_expansion and c2c_expansion=1 aren't compatible
            self.g.add_division(total_expansion=5)

    def test_add_division_1(self):
        """double grading, set start_size and c2c_expansion"""
        self.g.length = 2

        self.g.add_division(length_ratio=0.5, start_size=0.1, c2c_expansion=1.1)
        self.g.add_division(length_ratio=0.5, start_size=0.1, c2c_expansion=1.1, invert=True)

        self.assertListEqual(self.g.divisions, [[0.5, 8, 1.9487171000000012], [0.5, 8, 0.5131581182307065]])

    def test_add_division_2(self):
        """single grading, set c2c_expansion and count"""
        self.g.add_division(1, c2c_expansion=1.1, count=10)
        self.assertListEqual(self.g.divisions, [[1, 10, 2.357947691000002]])

    def test_add_division_3(self):
        """single grading, set count and start_size"""
        self.g.add_division(1, count=10, start_size=0.05)

        self.assertListEqual(self.g.divisions, [[1, 10, 3.433788027752166]])

    def test_is_defined(self):
        self.g.add_division(1, count=10, start_size=0.05)

        self.assertTrue(self.g.is_defined)

    def test_is_not_defined(self):
        self.assertFalse(self.g.is_defined)