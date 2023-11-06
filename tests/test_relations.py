# numbers are calculated with the calculator all this is 'borrowed' from
# https://openfoamwiki.net/index.php/Scripts/blockMesh_grading_calculation
# with a few differences:
# - scipy.optimize.<whatever> can be used here instead of barbarian bisection
# - all floats are converted to integers by rounding down (only matters for border cases)
import unittest

from parameterized import parameterized

from classy_blocks.grading import relations as rel
from classy_blocks.grading.chop import ChopRelation
from classy_blocks.util import functions as f


class TestGradingRelations(unittest.TestCase):
    """Testing valid, border and invalid cases"""

    def assertAlmostEqual(self, *args, **kwargs):  # noqa: N802
        kwargs.pop("places", None)

        kwargs["places"] = 5

        return super().assertAlmostEqual(*args, **kwargs)

    @parameterized.expand(
        (
            ((1, 10, 1), 0.1),
            ((1, 10, 1.1), 0.06274539488),
        )
    )
    def test_get_start_size__count__c2c_expansion_valid(self, args, result):
        self.assertAlmostEqual(rel.get_start_size__count__c2c_expansion(*args), result, places=5)

    def test_get_start_size__count__c2c_expansion_zerolen(self):
        with self.assertRaises(ValueError):
            rel.get_start_size__count__c2c_expansion(0, 10, 1)

    def test_get_start_size__count__c2c_expansion_contracting(self):
        with self.assertRaises(ValueError):
            rel.get_start_size__count__c2c_expansion(1, 0.5, 1)

    def test_get_start_size__end_size__total_expansion_valid(self):
        self.assertAlmostEqual(rel.get_start_size__end_size__total_expansion(1, 0.1, 1), 0.1)

    def test_get_start_size__end_size__total_expansion_zerolen(self):
        with self.assertRaises(ValueError):
            rel.get_start_size__end_size__total_expansion(0, 0.1, 1)

    def test_get_start_size__end_size__total_expansion_zeroexp(self):
        with self.assertRaises(ValueError):
            rel.get_start_size__end_size__total_expansion(1, 0.1, 0)

    def test_get_end_size__start_size__total_expansion_valid(self):
        self.assertAlmostEqual(rel.get_end_size__start_size__total_expansion(1, 0.1, 10), 1)

    def test_get_end_size__start_size__total_expansion_neglen(self):
        with self.assertRaises(ValueError):
            rel.get_end_size__start_size__total_expansion(-1, 0.1, 0)

    @parameterized.expand(
        (
            ((1, 1, 1), 2),
            ((1, 0.1, 1), 11),
            ((1, 0.1, 1.1), 8),
            ((1, 2, 1), 1),  # border cases
            ((1, 1, 2), 2),
        )
    )
    def test_get_count__start__size__c2c_expansion_valid(self, args, result):
        self.assertAlmostEqual(rel.get_count__start_size__c2c_expansion(*args), result)

    @parameterized.expand(
        (
            ((0, 0.1, 1.1),),
            ((1, 0, 1.1),),
            ((1, 0.95, 0),),
        )
    )
    def test_get_count__start__size__c2c_expansion_invalid(self, args):
        # invalid cases:
        # length < 0
        # start_size = 0
        # c2c_expansion = 0
        with self.assertRaises(ValueError):
            rel.get_count__start_size__c2c_expansion(*args)

    @parameterized.expand(
        (
            ((1, 0.1, 1), 11),
            ((1, 0.1, 1.1), 26),
            ((1, 0.1, 0.9), 8),
            ((1, 1, 1), 2),  # border cases
            ((1, 1, 2), 2),
        )
    )
    def test_get_count__end_size__c2c_expansion_valid(self, args, result):
        self.assertAlmostEqual(rel.get_count__end_size__c2c_expansion(*args), result)

    def test_get_count__end_size__c2c_expansion_invalid(self):
        with self.assertRaises(ValueError):
            rel.get_count__end_size__c2c_expansion(1, 0.1, 1.5)

    def test_get_count__total_expansion__c2c_expansion_valid(self):
        self.assertAlmostEqual(rel.get_count__total_expansion__c2c_expansion(1, 3, 1.1), 12)

    def test_get_count__total_expansion__c2c_expansion_border(self):
        self.assertAlmostEqual(rel.get_count__total_expansion__c2c_expansion(1, 1, 1.1), 1)

    @parameterized.expand(
        (
            ((1, 1, 1),),
            ((1, -1, 1.1),),
        )
    )
    def test_get_count__total_expansion__c2c_expansion_invalid(self, args):
        with self.assertRaises(ValueError):
            rel.get_count__total_expansion__c2c_expansion(*args)

    @parameterized.expand(
        (
            ((1, 1, 0.1), 10),
            ((1, 2, 0.1), 7),
            ((1, 8, 0.1), 3),
            ((1, 0.9, 0.5), 3),  # border cases
            ((1, 0.3, 1), 2),
        )
    )
    def test_get_count__total_expansion__start_size_valid(self, args, result):
        self.assertAlmostEqual(rel.get_count__total_expansion__start_size(*args), result)

    @parameterized.expand(
        (
            ((1, 10, 0.1), 1),
            ((1, 2, 0.1), 9),
            ((1, 5, 0.1), 1.352395572),
            ((1, 2, 0.5), 1),
            ((1, 10, 0.05), 1.1469127),
            ((1, 1, 0.1), 1),  # border cases
            ((1, 20, 0.1), 0.9181099911),
        )
    )
    def test_get_c2c_expansion__count__start_size_valid(self, args, result):
        self.assertAlmostEqual(rel.get_c2c_expansion__count__start_size(*args), result)

    @parameterized.expand(
        (
            ((0, 1, 0.1),),  # length = 0
            ((1, 0, 0.1),),  # count < 1
            ((1, 10, 1.1),),  # start_size > length
            ((1, 10, 0),),  # start_size = 0
            ((1, 10, 0.9),),
        )
    )
    def test_get_c2c_expansion__count__start_size_invalid(self, args):
        with self.assertRaises(ValueError):
            rel.get_c2c_expansion__count__start_size(*args)

    @parameterized.expand(
        (
            ((1, 10, 0.1), 1),
            ((1, 10, 0.01), 0.6784573173),
            ((1, 10, 0.2), 1.202420088),
            ((1, 1, 1), 1),  # border case
        )
    )
    def test_get_c2c_expansion__count__end_size_valid(self, args, result):
        self.assertAlmostEqual(rel.get_c2c_expansion__count__end_size(*args), result)

    @parameterized.expand(
        (
            ((1, 0.5, 1),),
            ((1, 10, -0.5),),
            ((1, 10, 1),),
        )
    )
    def test_get_c2c_expansion__count__end_size_invalid(self, args):
        with self.assertRaises(ValueError):
            rel.get_c2c_expansion__count__end_size(*args)

    @parameterized.expand(
        (
            ((1, 10, 5), 1.195813175),
            ((1, 10, 0.5), 0.9258747123),
            ((1, 10, 1), 1),  # border case
        )
    )
    def test_get_c2c_expansion__count__total_expansion_valid(self, args, result):
        self.assertAlmostEqual(rel.get_c2c_expansion__count__total_expansion(*args), result)

    def test_get_c2c_expansion__count__total_expansion_invalid(self):
        with self.assertRaises(ValueError):
            rel.get_c2c_expansion__count__total_expansion(1, 1, 1)

    @parameterized.expand((((1, 10, 1), 1), ((1, 1, 1), 1), ((1, 10, 1.1), 2.3579476), ((1, 1, 1), 1)))  # border case
    def test_get_total_expansion__count__c2c_expansion_valid(self, args, result):
        self.assertAlmostEqual(rel.get_total_expansion__count__c2c_expansion(*args), result)

    def test_get_total_expansion__count__c2c_expansion_invalid(self):
        with self.assertRaises(ValueError):
            rel.get_total_expansion__count__c2c_expansion(1, 0.5, 1)

    @parameterized.expand(
        (
            ((1, 1, 1), 1),
            ((1, 0.1, 0.01), 0.1),
            ((1, 0.1, 0.01), 0.1),
            ((1, 0.01, 0.1), 10),
        )
    )
    def test_get_total_expansion__start_size__end_size_valid(self, args, result):
        self.assertAlmostEqual(rel.get_total_expansion__start_size__end_size(*args), result)

    def test_get_total_expansion__start_size__end_size_invalid(self):
        with self.assertRaises(ValueError):
            rel.get_total_expansion__start_size__end_size(1, 0, 0.1)

    @parameterized.expand(
        (
            (0.5, ">0"),
            (1, ">=0"),
            (1, ">=1"),
            (1.5, ">1"),
            (1.5, "<2"),
            (2, "<=2"),
            (2, "==2"),
            (3, "!=2"),
        )
    )
    def test_validate_count_valid(self, count, condition):
        rel._validate_count(count, condition)

    def test_validate_count_invalid_type(self):
        with self.assertRaises(TypeError):
            rel._validate_count("a", "==")

    def test_validate_count_invalid_operator(self):
        with self.assertRaises(ValueError):
            rel._validate_count(10, "xx")

    def test_validate_count_unknown_operator(self):
        with self.assertRaises(ValueError):
            rel._validate_count(10, ">x")


class ChopRelationTests(unittest.TestCase):
    def test_from_function_invalid(self):
        """Raise an exception when an unknown relation is found"""
        with self.assertRaises(RuntimeError):
            _ = ChopRelation.from_function(f.angle_between)
