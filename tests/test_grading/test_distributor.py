import unittest

import numpy as np

from classy_blocks.grading.autograding.params.distributor import SmoothDistributor


class SmoothDistributorTests(unittest.TestCase):
    # Tests for proof-of-concept and development hacks;
    # TODO: provide assertions, not just prints

    def test_raw_simple(self):
        smoother = SmoothDistributor(10, 0.1, 1, 0.1)

        np.testing.assert_almost_equal(smoother.get_raw_coords(), np.arange(-0.1, 1.1, 0.1))

    def test_get_smooth_simple(self):
        # just to make sure it runs
        smoother = SmoothDistributor(10, 0.1, 1, 0.1)

        np.testing.assert_almost_equal(smoother.get_smooth_coords(), np.arange(-0.1, 1.1, 0.1), decimal=5)

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
