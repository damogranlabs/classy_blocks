
import copy
from classy_blocks.data.point import Point
from classy_blocks.items.vertex import Vertex
from classy_blocks.util.pair import Pair

from tests.fixtures import FixturedTestCase

class PairTests(FixturedTestCase):
    """Tests of Pair object"""
    def setUp(self):
        block = self.get_blocks()[0]

        self.vertices = [Vertex(Point(block.points[i]), i) for i in range(8)]
        self.axis = 0
        self.index = 0
    
    @property
    def pair(self) -> Pair:
        """The test subject"""
        return Pair(self.vertices, self.axis, self.index)
    
    def test_assert_vertices(self):
        """Check length of vertices"""
        with self.assertRaises(AssertionError):
            del self.vertices[0]
            _ = self.pair
        
    def test_assert_axis(self):
        """Check axis index"""
        with self.assertRaises(AssertionError):
            self.axis = -1
            _ = self.pair

        with self.assertRaises(AssertionError):
            self.axis = 4
            _ = self.pair

    def test_assert_index(self):
        """Check pair index"""
        with self.assertRaises(AssertionError):
            self.index = -1
            _ = self.pair

        with self.assertRaises(AssertionError):
            self.index = 4
            _ = self.pair

    def test_is_valid(self):
        """Valid with two different vertices"""
        self.assertTrue(self.pair.is_valid)

    def test_is_invalid(self):
        """Not valid with two same vertices"""
        self.vertices[1] = self.vertices[0]
        self.assertFalse(self.pair.is_valid)

    def test_coincident_aligned(self):
        """Coincident pair (__eq__()) with an aligned pair"""
        pair_1 = self.pair
        pair_2 = copy.copy(self.pair)

        self.assertTrue(pair_1 == pair_2)
    
    def test_coincident_invertex(self):
        """Coincident pair (__eq__()) with an inverted pair"""
        pair_1 = self.pair
        pair_2 = copy.copy(self.pair)

        # invert the other one
        pair_2.vertex_1, pair_2.vertex_2 = pair_2.vertex_2, pair_2.vertex_1

        self.assertTrue(pair_1 == pair_2)
    
    def test_not_coincident(self):
        """Non-coincident pair"""
        pair_1 = self.pair
        self.index = 1
        pair_2 = self.pair

        self.assertFalse(pair_1 == pair_2)

    def test_is_aligned_exception(self):
        """Raise an exception if pairs are not equal"""
        pair_1 = self.pair

        self.index = 1
        pair_2 = self.pair

        with self.assertRaises(RuntimeError):
            pair_1.is_aligned(pair_2)

    def test_is_aligned(self):
        """Alignment: the same"""
        pair_1 = self.pair
        pair_2 = copy.copy(self.pair)

        self.assertTrue(pair_1.is_aligned(pair_2))

    def test_is_invertex(self):
        """Alignment: opposite"""
        pair_1 = self.pair
        pair_2 = copy.copy(self.pair)

        pair_2.vertex_1, pair_2.vertex_2 = pair_2.vertex_2, pair_2.vertex_1

        self.assertFalse(pair_1.is_aligned(pair_2))