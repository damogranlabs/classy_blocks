import unittest

from parameterized import parameterized

from classy_blocks.util import tools
from classy_blocks.base.exceptions import CornerPairError

class ToolsTests(unittest.TestCase):
    @parameterized.expand((
        (0, 2),
        (1, 3),
        (1, 6),
        (1, 7),
    ))
    def test_wrong_edge_location(self, corner_1, corner_2):
        """Raise an exception when an invalid corner pair is given"""
        edge_location = tools.EdgeLocation(corner_1, corner_2, "top")

        with self.assertRaises(CornerPairError):
            _ = edge_location.start_corner