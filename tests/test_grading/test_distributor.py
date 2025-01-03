import unittest

import numpy as np

from classy_blocks.grading.autograding.inflation.distributor import DoubleInflationDistributor, InflationDistributor
from classy_blocks.grading.autograding.smooth.distributor import SmoothDistributor


class SmoothDistributorTests(unittest.TestCase):
    # Tests for proof-of-concept and development hacks;
    # TODO: provide assertions, not just prints

    def test_raw_simple(self):
        smoother = SmoothDistributor(10, 0.1, 1, 0.1)

        np.testing.assert_almost_equal(smoother.get_raw_coords(), np.arange(-0.1, 1.1, 0.1))

    def test_get_smooth_simple(self):
        # just to make sure it runs
        smoother = SmoothDistributor(10, 0.1, 1, 0.1)

        np.testing.assert_almost_equal(smoother.get_smooth_coords(), np.arange(0, 1.1, 0.1), decimal=5)

    def test_raw_variable(self):
        smoother = SmoothDistributor(10, 0.1, 1, 0.01)

        expected = np.concatenate((np.linspace(-0.1, 1, num=12), [1.01]))

        np.testing.assert_almost_equal(smoother.get_raw_coords(), expected)

    def test_smooth_variable1(self):
        smoother = SmoothDistributor(10, 0.1, 1, 0.01)

        _ = smoother.get_smooth_coords()

    def test_smooth_variable2(self):
        smoother = SmoothDistributor(20, 0.01, 1, 0.01)

        coords = smoother.get_smooth_coords()
        print(coords)

    def test_get_chops(self):
        smoother = SmoothDistributor(20, 0.01, 1, 0.01)

        print(smoother.get_chops(3))


class InflationDistributorTests(unittest.TestCase):
    def setUp(self):
        self.distributor = InflationDistributor(20, 1e-3, 1, 0.1, 1.2, 30, 2, 0.1)

    def test_ideal_ratios(self):
        ratios = self.distributor.get_ideal_ratios()

        exp_count = len(np.where(ratios == 1.2)[0])
        self.assertEqual(exp_count, 11)

        buffer_count = len(np.where(ratios == 2)[0])
        self.assertEqual(buffer_count, 4)

        bulk_count = len(np.where(ratios == 1)[0])
        self.assertEqual(bulk_count, 20 - exp_count - buffer_count + 1)

    def test_flip_ratios(self):
        ratios = self.distributor.get_ideal_ratios()

        print(DoubleInflationDistributor.flip_ratios(ratios))
