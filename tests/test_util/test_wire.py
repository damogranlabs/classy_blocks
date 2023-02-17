import copy

from classy_blocks.data.point import Point
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.wire import Wire

from tests.fixtures import FixturedTestCase

class WireTests(FixturedTestCase):
    """Tests of Pair object"""
    def setUp(self):
        block = self.get_blocks()[0]

        self.vertices = [Vertex(Point(block.points[i]), i) for i in range(8)]

        self.corner_1 = 1
        self.corner_2 = 2
        self.axis = 1 # make sure corners and axis are consistent
    
    @property
    def wire(self) -> Wire:
        """The test subject"""
        return Wire(self.vertices, self.axis, self.corner_1, self.corner_2)
    
    def test_is_valid(self):
        """Valid with two different vertices"""
        self.assertTrue(self.wire.is_valid)

    def test_is_invalid(self):
        """Not valid with two same vertices"""
        self.vertices[1] = self.vertices[0]
        self.assertFalse(self.wire.is_valid)

    def test_coincident_aligned(self):
        """Coincident pair (__eq__()) with an aligned pair"""
        pair_1 = self.wire
        pair_2 = copy.copy(self.wire)

        self.assertTrue(pair_1 == pair_2)
    
    def test_coincident_invertex(self):
        """Coincident pair (__eq__()) with an inverted pair"""
        pair_1 = self.wire
        pair_2 = copy.copy(self.wire)

        # invert the other one
        pair_2.vertex_1, pair_2.vertex_2 = pair_2.vertex_2, pair_2.vertex_1

        self.assertTrue(pair_1 == pair_2)
    
    def test_not_coincident(self):
        """Non-coincident pair"""
        pair_1 = self.wire
        self.index = 1
        pair_2 = self.wire

        self.assertFalse(pair_1 == pair_2)

    def test_is_aligned_exception(self):
        """Raise an exception if pairs are not equal"""
        pair_1 = self.wire

        self.index = 1
        pair_2 = self.wire

        with self.assertRaises(RuntimeError):
            pair_1.is_aligned(pair_2)

    def test_is_aligned(self):
        """Alignment: the same"""
        pair_1 = self.wire
        pair_2 = copy.copy(self.wire)

        self.assertTrue(pair_1.is_aligned(pair_2))

    def test_is_invertex(self):
        """Alignment: opposite"""
        pair_1 = self.wire
        pair_2 = copy.copy(self.wire)

        pair_2.vertex_1, pair_2.vertex_2 = pair_2.vertex_2, pair_2.vertex_1

        self.assertFalse(pair_1.is_aligned(pair_2))