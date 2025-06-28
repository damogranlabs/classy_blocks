import unittest

import numpy as np
from parameterized import parameterized

from classy_blocks.optimize import quality


class QualityTests(unittest.TestCase):
    @parameterized.expand(
        (
            ([1.0, 1.0, 0.0], 90),
            ([0.5, 0.5, 0.0], 180),
            ([0.0, 0.0, 0.0], 270),
        )
    )
    def test_quad_inner_angle(self, point_2, angle):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], point_2, [0.0, 1.0, 0.0]])
        normal = np.array([0.0, 0.0, 1.0])

        result = quality.get_quad_inner_angle(points, normal, 2)

        self.assertGreater(result, angle - 1)
        self.assertLess(result, angle + 1)

    def test_quad_quality(self):
        # compare a perfect quad with a little-less-than-perfect
        # and make sure calculated quality of the latter is worse
        grid_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.1, 1.1, 0]])

        quality_1 = quality.get_quad_quality(grid_points, np.array([0, 1, 2, 3]))
        quality_2 = quality.get_quad_quality(grid_points, np.array([0, 1, 4, 3]))

        self.assertGreater(quality_2, quality_1)
