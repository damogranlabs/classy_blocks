import copy

from classy_blocks.base.exceptions import InconsistentGradingsError
from classy_blocks.cbtyping import DirectionType
from classy_blocks.construct.edges import Arc
from classy_blocks.grading.chop import Chop
from classy_blocks.items.edges.arcs.arc import ArcEdge
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

    def test_is_valid(self):
        """Valid with two different vertices"""
        self.assertTrue(self.wire.is_valid)

    def test_is_invalid(self):
        """Not valid with two same vertices"""
        self.vertices[2] = self.vertices[1]
        self.assertFalse(self.wire.is_valid)

    def test_coincident_aligned(self):
        """Coincident pair (__eq__()) with an aligned pair"""
        wire_1 = self.wire
        wire_2 = copy.copy(self.wire)

        self.assertTrue(wire_1.is_coincident(wire_2))

    def test_coincident_inverted(self):
        """Coincident pair (__eq__()) with an inverted pair"""
        wire_1 = self.wire
        wire_2 = copy.copy(self.wire)

        # invert the other one
        wire_2.vertices.reverse()

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
        wire_2 = copy.copy(self.wire)

        wire_2.vertices.reverse()

        self.assertFalse(wire_1.is_aligned(wire_2))

    def test_add_inline_duplicate(self):
        wire = self.wire

        wire.add_inline(wire)
        self.assertEqual(len(wire.after), 0)
        self.assertEqual(len(wire.before), 0)

    def test_check_consistency_count(self):
        wire_1 = Wire(self.vertices, 0, 0, 1)
        wire_1.grading.add_chop(Chop(count=10))
        wire_1.update()

        wire_2 = Wire(self.vertices, 0, 0, 1)
        wire_2.grading.add_chop(Chop(count=5))
        wire_2.update()

        wire_1.coincidents.add(wire_2)

        with self.assertRaises(InconsistentGradingsError):
            wire_1.check_consistency()

    def test_check_consistency_length(self):
        # Add two different edges so that lengths are not equal
        wire_1 = Wire(self.vertices, 0, 0, 1)
        wire_1.edge = ArcEdge(self.vertices[0], self.vertices[1], Arc([0.5, 0.5, 0]))

        wire_2 = Wire(self.vertices, 0, 0, 1)
        wire_2.edge = ArcEdge(self.vertices[0], self.vertices[1], Arc([0.5, 0.25, 0]))
        wire_2.update()

        wire_1.add_coincident(wire_2)

        with self.assertRaises(InconsistentGradingsError):
            wire_1.check_consistency()
