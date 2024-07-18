import unittest

import numpy as np
from parameterized import parameterized

from classy_blocks.construct.array import Array
from classy_blocks.construct.curves.interpolators import LinearInterpolator


class LinearInterpolatorTests(unittest.TestCase):
    def setUp(self):
        # a simple square wave
        self.points = Array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [1, 0, 0],
                [2, 0, 0],
            ]
        )

    @parameterized.expand(
        (
            (0, [0, 0, 0]),
            (1 / 4, [0, 1, 0]),
            (2 / 4, [1, 1, 0]),
            (0.5 / 4, [0, 0.5, 0]),
        )
    )
    def test_points(self, param, result):
        intfun = LinearInterpolator(self.points, True)

        np.testing.assert_almost_equal(intfun(param), result)

    def test_extrapolate_exception(self):
        intfun = LinearInterpolator(self.points, False)

        with self.assertRaises(ValueError):
            _ = intfun(-1)
