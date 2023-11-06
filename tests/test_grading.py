import unittest

from parameterized import parameterized

from classy_blocks.grading import relations as rel
from classy_blocks.grading.chop import Chop, ChopRelation
from classy_blocks.grading.grading import Grading


class TestGrading(unittest.TestCase):
    def setUp(self):
        self.g = Grading(1)

    def test_calculator_functions(self):
        expected_functions = [
            # return_value | param1 | param2 (param0 = length)
            ["c2c_expansion", ["count", "end_size"], rel.get_c2c_expansion__count__end_size],
            ["c2c_expansion", ["count", "start_size"], rel.get_c2c_expansion__count__start_size],
            ["c2c_expansion", ["count", "total_expansion"], rel.get_c2c_expansion__count__total_expansion],
            ["count", ["end_size", "c2c_expansion"], rel.get_count__end_size__c2c_expansion],
            ["count", ["start_size", "c2c_expansion"], rel.get_count__start_size__c2c_expansion],
            ["count", ["total_expansion", "c2c_expansion"], rel.get_count__total_expansion__c2c_expansion],
            ["count", ["total_expansion", "start_size"], rel.get_count__total_expansion__start_size],
            ["end_size", ["start_size", "total_expansion"], rel.get_end_size__start_size__total_expansion],
            ["start_size", ["count", "c2c_expansion"], rel.get_start_size__count__c2c_expansion],
            ["start_size", ["end_size", "total_expansion"], rel.get_start_size__end_size__total_expansion],
            ["total_expansion", ["count", "c2c_expansion"], rel.get_total_expansion__count__c2c_expansion],
            ["total_expansion", ["start_size", "end_size"], rel.get_total_expansion__start_size__end_size],
        ]
        expected_functions = [ChopRelation(f[0], f[1][0], f[1][1], f[2]) for f in expected_functions]

        self.assertCountEqual(expected_functions, ChopRelation.get_possible_combinations())

    @parameterized.expand(
        (
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
        )
    )
    def test_calculate(self, keys, count, total_expansion):
        chop = Chop(1, **keys)

        self.assertAlmostEqual(chop.calculate(1)[0], count)
        self.assertAlmostEqual(chop.calculate(1)[1], total_expansion, places=5)

    def add_division(self, length_ratio, count_ratio, total_expansion):
        self.g.specification.append([length_ratio, count_ratio, total_expansion])

    def test_output_empty(self):
        with self.assertRaises(ValueError):
            _ = self.g.description

    def test_output_single(self):
        self.add_division(1, 1, 3)
        self.assertEqual(str(self.g.description), "3")

    def test_output_multi(self):
        self.add_division(0.25, 0.4, 2)
        self.add_division(0.5, 0.2, 1)
        self.add_division(0.25, 0.4, 0.5)

        expected_output = "((0.25 0.4 2)(0.5 0.2 1)(0.25 0.4 0.5))"

        self.assertEqual(str(self.g.description), expected_output)

    def test_copy_invert_simple(self):
        self.add_division(1, 1, 5)

        self.assertAlmostEqual(self.g.specification[0][2], 5)
        self.assertAlmostEqual(self.g.inverted.specification[0][2], 0.2)

    def test_add_division_zero_length(self):
        """Add a chop to zero-length grading"""
        with self.assertRaises(ValueError):
            self.g.length = 0
            self.g.add_chop(Chop(count=10))

    def test_insuficient_data(self):
        """Add a chop with not enough data to calculate grading"""
        self.g.length = 1
        with self.assertRaises(ValueError):
            # when using only 1 parameter, c2c_expansion is assumed 1;
            # when specifying that as well, another parameter must be provided
            self.g.add_chop(Chop(c2c_expansion=1.1))

    def test_wrong_combination(self):
        """Add a chop with specified total_ and c2c_expansion"""
        with self.assertRaises(ValueError):
            # specified total_expansion and c2c_expansion=1 aren't compatible
            self.g.add_chop(Chop(total_expansion=5))

    def test_add_division_1(self):
        """double grading, set start_size and c2c_expansion"""
        self.g.length = 2

        self.g.add_chop(Chop(length_ratio=0.5, start_size=0.1, c2c_expansion=1.1))
        self.g.add_chop(Chop(length_ratio=0.5, start_size=0.1, c2c_expansion=1.1, invert=True))

        self.assertListEqual(self.g.specification, [[0.5, 8, 1.9487171000000012], [0.5, 8, 0.5131581182307065]])

    def test_add_division_2(self):
        """single grading, set c2c_expansion and count"""
        self.g.add_chop(Chop(1, c2c_expansion=1.1, count=10))
        self.assertListEqual(self.g.specification, [[1, 10, 2.357947691000002]])

    def test_add_division_3(self):
        """single grading, set count and start_size"""
        self.g.add_chop(Chop(1, count=10, start_size=0.05))

        self.assertListEqual(self.g.specification, [[1, 10, 3.433788027752166]])

    def test_add_division_inverted(self):
        """Inverted chop, different result"""
        self.g.add_chop(Chop(0.5, count=10, start_size=0.05, invert=False))
        self.g.add_chop(Chop(0.5, count=10, start_size=0.05, invert=True))

        self.assertAlmostEqual(self.g.specification[0][2], 1 / self.g.specification[1][2])

    def test_add_wrong_ratio(self):
        """Add a chop with an invalid length ratio"""
        with self.assertRaises(ValueError):
            self.g.add_chop(Chop(length_ratio=0, count=10))

    def test_is_defined(self):
        self.g.add_chop(Chop(1, count=10, start_size=0.05))

        self.assertTrue(self.g.is_defined)

    def test_is_not_defined(self):
        self.assertFalse(self.g.is_defined)

    def test_warn_ratio(self):
        """Issue a warning when length_ratios don't add up to 1"""
        self.g.add_chop(Chop(length_ratio=0.6, start_size=0.1))
        self.g.add_chop(Chop(length_ratio=0.6, start_size=0.1))

        with self.assertWarns(Warning):
            _ = self.g.description

    def test_invert_empty(self):
        """Invert a grading with no chops"""
        self.assertEqual(id(self.g), id(self.g.inverted))

    def test_equal(self):
        """Two different gradings with same parameters are equal"""
        grad1 = Grading(1)
        grad2 = Grading(1)

        for g in (grad1, grad2):
            g.add_chop(Chop(length_ratio=0.5, start_size=0.1))
            g.add_chop(Chop(length_ratio=0.5, end_size=0.1))

        self.assertTrue(grad1 == grad2)

    def test_not_equal_divisionsn(self):
        """Two gradings with different lengths of specification"""
        grad1 = Grading(1)
        grad2 = Grading(1)

        for g in (grad1, grad2):
            g.add_chop(Chop(length_ratio=0.5, start_size=0.1))
            g.add_chop(Chop(length_ratio=0.5, end_size=0.1))

        grad1.add_chop(Chop(count=10))

        self.assertFalse(grad1 == grad2)

    def test_not_equal(self):
        """Two gradings with equal lengths of specification"""
        grad1 = Grading(1)
        grad1.add_chop(Chop(length_ratio=0.5, start_size=0.15))

        grad2 = Grading(1)
        grad2.add_chop(Chop(length_ratio=0.5, end_size=0.1))

        self.assertFalse(grad1 == grad2)
