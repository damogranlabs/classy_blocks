import unittest

import numpy as np

from classy_blocks.grading.autograding.params.smoother import Smoother


class SmootherTests(unittest.TestCase):
    def test_raw_simple(self):
        smoother = Smoother(10, 0.1, 1, 0.1)

        np.testing.assert_almost_equal(smoother.get_raw_coords(), np.arange(-0.1, 1.1, 0.1))

    def test_get_smooth_simple(self):
        # just to make sure it runs
        smoother = Smoother(10, 0.1, 1, 0.1)

        np.testing.assert_almost_equal(smoother.get_smooth_coords(), np.arange(-0.1, 1.1, 0.1), decimal=5)

    def test_raw_variable(self):
        smoother = Smoother(10, 0.1, 1, 0.01)

        expected = np.concatenate((np.linspace(-0.1, 1, num=12), [1.01]))

        np.testing.assert_almost_equal(smoother.get_raw_coords(), expected)

    def test_smooth_variable1(self):
        smoother = Smoother(10, 0.1, 1, 0.01)

        _ = smoother.get_smooth_coords()

    def test_smooth_variable2(self):
        smoother = Smoother(20, 0.01, 1, 0.01)

        coords = smoother.get_smooth_coords()
        print(coords)

    def test_get_chops(self):
        smoother = Smoother(20, 0.01, 1, 0.01)

        print(smoother.get_chops(3))
