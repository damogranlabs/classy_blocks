import unittest
from parameterized import parameterized

from wire import Wireframe
from wire import Wire

class WireframeTests(unittest.TestCase):
    def setUp(self):
        self.vertices = [0, 1, 2, 3, 4, 5, 6, 7]

    @property
    def wf(self) -> Wireframe:
        """The test subject"""
        return Wireframe(self.vertices)

    def test_init(self):
        """Generate 8 Wire objects"""
        self.assertEqual(len(self.wf.wires), 12)
    
    def test_neighbour_count(self):
        """Each corner has exactly 3 neighbours"""
        for neighbours in self.wf.neighbours:
            self.assertEqual(len(neighbours), 3)

    def test_neighbour_indexes(self):
        """Check that indexes are generated exactly as the sketch dictates"""
        # hand-typed
        expected_indexes = (
            (1, 3, 4), # 0
            (0, 2, 5), # 1
            (1, 3, 6), # 2
            (0, 2, 7), # 3
            (0, 5, 7), # 4
            (1, 4, 6), # 5
            (2, 5, 7), # 6
            (3, 4, 6), # 7
        )

        for i, neighbours in enumerate(self.wf.neighbours):
            generated = set(neighbours.keys())
            expected = set(expected_indexes[i])

            self.assertSetEqual(generated, expected)

    def test_axis_count(self):
        """Check that each axis has exactly 4 wires"""
        for axis in range(3):
            self.assertEqual(len(self.wf.axes[axis]), 4)

    @parameterized.expand(((0, 2), (1, 3), (0, 6), (1, 7)))
    def test_find_wire_fail(self, corner_1, corner_2):
        """There are no wires in face or volume diagonals"""
        with self.assertRaises(KeyError):
            _ = self.wf[corner_1][corner_2]

    @parameterized.expand(((0, 1), (1, 0), (0, 4), (4, 0)))
    def test_find_wire_success(self, corner_1, corner_2):
        """Find wire the old school way"""
        self.assertEqual(type(self.wf[corner_1][corner_2]), Wire)