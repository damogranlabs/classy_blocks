import copy

from classy_blocks.cbtyping import DirectionType
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.wires.wire import Wire
from tests.fixtures.data import DataTestCase


class WireTests(DataTestCase):
    """Tests of Pair object"""

    def setUp(self) -> None:
        block = self.get_single_data(0)

        self.vertices = [Vertex(block.points[i], i) for i in range(8)]

        self.corner_1 = 1
        self.corner_2 = 2
        self.axis: DirectionType = 1  # make sure corners and axis are consistent

    @property
    def wire(self) -> Wire:
        """The test subject"""
        return Wire(self.vertices, self.axis, self.corner_1, self.corner_2)

    @property
    def inverted_wire(self) -> Wire:
        """The test subject, pointing the other way"""
        return Wire(self.vertices, self.axis, self.corner_2, self.corner_1)

    def test_coincident_aligned(self):
        """Coincident pair (__eq__()) with an aligned pair"""
        wire_1 = self.wire
        wire_2 = copy.copy(self.wire)

        self.assertTrue(wire_1.is_coincident(wire_2))

    def test_coincident_inverted(self):
        """Coincident pair (__eq__()) with an inverted pair"""
        wire_1 = self.wire
        wire_2 = self.inverted_wire

        self.assertTrue(wire_1.is_coincident(wire_2))

    def test_not_coincident(self):
        """Non-coincident pair"""
        wire_1 = self.wire

        self.corner_1 = 0
        self.corner_2 = 1
        self.axis = 0
        wire_2 = self.wire

        self.assertFalse(wire_1 == wire_2)

    def test_is_aligned_exception(self):
        """Raise an exception if pairs are not equal"""
        wire_1 = self.wire

        self.corner_1 = 0
        self.corner_2 = 1
        self.axis = 0
        wire_2 = self.wire

        self.assertFalse(wire_1 == wire_2)

        with self.assertRaises(RuntimeError):
            wire_1.is_aligned(wire_2)

    def test_is_aligned(self):
        """Alignment: the same"""
        wire_1 = self.wire
        wire_2 = copy.copy(self.wire)

        self.assertTrue(wire_1.is_aligned(wire_2))

    def test_is_inverted(self):
        """Alignment: opposite"""
        wire_1 = self.wire
        wire_2 = self.inverted_wire

        self.assertFalse(wire_1.is_aligned(wire_2))
