import unittest

import numpy as np
from parameterized import parameterized

from classy_blocks.base.exceptions import AnnulusCreationError
from classy_blocks.construct.flat.sketches.annulus import Annulus


class AnnulusTests(unittest.TestCase):
    """Critical stuff on Annuli"""

    def test_invalid_inner_radius(self):
        with self.assertRaises(AnnulusCreationError):
            Annulus([2, 2, 2], [3, 3, 2], [0, 0, 1], 2)  # (3, 3) - (2, 2) = 1.414.., which is < 2

    def test_coplanar_assert(self):
        """Assert the defining points produce a planar annulus or
        the 'center' will be calculated differently than it was defined"""
        with self.assertRaises(AnnulusCreationError):
            Annulus([2, 2, 2], [0, 0, 0], [0, 0, 1], 0.5, 5)

    @parameterized.expand(((3,), (4,), (5,), (6,), (7,), (8,), (9,), (16,)))
    def test_center(self, n_segments):
        """Test that center is calculated back exactly as it was defined"""
        center = [2.0, 2.0, 0.0]
        ann = Annulus(center, [3, 1, 0], [1, 1, 0], 0.5, n_segments)

        np.testing.assert_array_almost_equal(center, ann.center)
