import unittest

from parameterized import parameterized
import numpy as np

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
        np.testing.assert_array_almost_equal(box.points[6] - box.points[0], [1, 1, 1])
