import unittest

import numpy as np
from parameterized import parameterized

from classy_blocks.construct.operations.box import Box


class BoxTests(unittest.TestCase):
    """Creation of boxes from all 8 diagonals"""

    @parameterized.expand(
        (
            (
                [
                    1,
                    1,
                    1,
                ],
            ),
            (
                [
                    -1,
                    1,
                    1,
                ],
            ),
            (
                [
                    -1,
                    -1,
                    1,
                ],
            ),
            (
                [
                    1,
                    -1,
                    1,
                ],
            ),
            (
                [
                    1,
                    1,
                    -1,
                ],
            ),
            (
                [
                    -1,
                    1,
                    -1,
                ],
            ),
            (
                [
                    -1,
                    -1,
                    -1,
                ],
            ),
            (
                [
                    1,
                    -1,
                    -1,
                ],
            ),
        )
    )
    def test_create_box(self, diagonal_point):
        """Create a box from an arbitrary set of diagonal points"""
        box = Box([0.0, 0.0, 0.0], diagonal_point)

        # the diagonal must be the same in all cases
        np.testing.assert_array_almost_equal(box.point_array[6] - box.point_array[0], [1, 1, 1])
