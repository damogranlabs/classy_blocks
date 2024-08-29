from typing import List

import numpy as np
from parameterized import parameterized

from classy_blocks.construct.edges import Arc
from classy_blocks.grading.chop import Chop
from classy_blocks.items.block import Block
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.wires.wire import Wire
from classy_blocks.util import functions as f
from tests.fixtures.block import BlockTestCase


class BlockTests(BlockTestCase):
    """Block item tests"""

    @parameterized.expand(
        (
            ((1, 2), (0, 3)),  # corner_1, corner_2 of block_0 and corner_1, corner_2 of block_1
            ((5, 6), (4, 7)),
            ((1, 5), (0, 4)),
            ((2, 6), (3, 7)),
        )
    )
    def test_add_neighbour_1_wires(self, this_corners, nei_corners):
        """Two blocks that share a 'side' a.k.a. face"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.add_neighbour(block_1)

        # these two blocks share the whole face (1 2 6 5)
        # 4 wires altogether
        self.assertEqual(
            block_0.wires[this_corners[0]][this_corners[1]].coincidents, {block_1.wires[nei_corners[0]][nei_corners[1]]}
        )

    def test_add_neighbour_1_axes(self):
        """Two blocks that share a 'side' a.k.a. face"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.add_neighbour(block_1)

        # block_1 is block_0's neighbour in axes 1 and 2
        self.assertSetEqual(block_0.axes[0].neighbours, set())
        self.assertSetEqual(block_0.axes[1].neighbours, {block_1.axes[1]})
        self.assertSetEqual(block_0.axes[2].neighbours, {block_1.axes[2]})

    def test_add_neighbour_2_wires(self):
        """Two blocks that share an edge only"""
        block_0 = self.make_block(0)
        block_2 = self.make_block(2)

        block_0.add_neighbour(block_2)

        # there must be only 1 wire that only has 1 neighbour
        self.assertEqual(block_0.wires[2][6].coincidents, {block_2.wires[0][4]})

    def test_add_neighbour_2_axes(self):
        """Two blocks that share an edge only"""
        block_0 = self.make_block(0)
        block_2 = self.make_block(2)

        block_0.add_neighbour(block_2)
        # block_2 is block_0'2 neighbour only on axis 2
        self.assertSetEqual(block_0.axes[0].neighbours, set())
        self.assertSetEqual(block_0.axes[1].neighbours, set())
        self.assertSetEqual(block_0.axes[2].neighbours, {block_2.axes[2]})

    def test_add_neighbour_3(self):
        """Where three blocks meet, there's a wire with 2 coincident wires"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)
        block_2 = self.make_block(2)

        block_0.add_neighbour(block_1)
        block_0.add_neighbour(block_2)

        self.assertEqual(block_0.wires[2][6].coincidents, {block_1.wires[3][7], block_2.wires[0][4]})

    def test_add_neighbour_twice(self):
        """Add the same neighbour twice"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.add_neighbour(block_1)
        block_0.add_neighbour(block_1)

        self.assertEqual(len(block_0.axes[0].neighbours), 0)
        self.assertEqual(len(block_0.axes[1].neighbours), 1)
        self.assertEqual(len(block_0.axes[2].neighbours), 1)

    def test_add_self(self):
        """Add the same block as a neighbour"""
        block_0 = self.make_block(0)

        block_0.add_neighbour(block_0)

        self.assertEqual(len(block_0.axes[0].neighbours), 0)
        self.assertEqual(len(block_0.axes[1].neighbours), 0)
        self.assertEqual(len(block_0.axes[2].neighbours), 0)

    def test_add_neighbour_inverted(self):
        """Add a neighbour with an inverted axis"""
        data = self.get_single_data(0)
        points = data.points
        indexes = [7, 6, 5, 4, 3, 2, 1, 0]
        vertices = [Vertex(p, indexes[i]) for i, p in enumerate(points)]
        block_0 = Block(0, vertices)

        block_1 = self.make_block(1)

        block_0.add_neighbour(block_1)

        self.assertEqual(len(block_0.axes[0].neighbours), 0)
        self.assertEqual(len(block_0.axes[1].neighbours), 1)
        self.assertEqual(len(block_0.axes[2].neighbours), 1)

        self.assertFalse(block_0.axes[1].is_aligned(block_1.axes[1]))
        self.assertFalse(block_0.axes[2].is_aligned(block_1.axes[2]))

    def test_add_foreign(self):
        """Try to add a block that is not in contact
        as a neighbour"""
        block_0 = self.make_block(0)

        data = self.get_single_data(1)
        # displace points
        points = [np.array(p) + f.vector(10, 10, 10) for p in data.points]
        indexes = [8, 9, 10, 11, 12, 13, 14, 15]
        vertices = [Vertex(p, indexes[i]) for i, p in enumerate(points)]
        block_1 = Block(0, vertices)

        block_0.add_neighbour(block_1)

        self.assertEqual(len(block_0.axes[0].neighbours), 0)
        self.assertEqual(len(block_0.axes[1].neighbours), 0)
        self.assertEqual(len(block_0.axes[2].neighbours), 0)


class BlockSimpleGradingTests(BlockTestCase):
    """Tests of neighbours, copying grading and whatnot simple variant;
    edge grading is tested with use of Loft operations and so on, that is in test_edge_grading"""

    def test_is_defined_0(self):
        """block.is_defined with no gradings"""
        block_0 = self.make_block(0)
        self.assertFalse(block_0.is_defined)

    def test_is_defined_1(self):
        """block.is_defined with 1 grading"""
        block_0 = self.make_block(0)
        block_0.chop(0, Chop(count=10))

        self.assertFalse(block_0.is_defined)

    def test_is_defined_2(self):
        """block.is_defined with 2 gradings"""
        block_0 = self.make_block(0)
        block_0.chop(0, Chop(count=10))
        block_0.chop(1, Chop(count=10))

        self.assertFalse(block_0.is_defined)

    def test_is_defined_3(self):
        """block.is_defined with 3 gradings"""
        block_0 = self.make_block(0)
        block_0.chop(0, Chop(count=10))
        block_0.chop(1, Chop(count=10))
        block_0.chop(2, Chop(count=10))
        block_0.grade()

        self.assertTrue(block_0.is_defined)

    def test_is_defined_4(self):
        """block.is_defined with 4 gradings"""
        block_0 = self.make_block(0)
        block_0.chop(0, Chop(count=10))
        block_0.chop(0, Chop(count=10))
        block_0.chop(1, Chop(count=10))
        block_0.chop(2, Chop(count=10))
        block_0.grade()

        self.assertTrue(block_0.is_defined)

    def test_copy_grading_defined(self):
        """block.copy_grading when it's already defined"""
        block_0 = self.make_block(0)
        block_0.chop(0, Chop(count=10))
        block_0.chop(1, Chop(count=10))
        block_0.chop(2, Chop(count=10))

        self.assertFalse(block_0.copy_grading())

    def test_copy_grading_undefined(self):
        """block.copy_grading when it's not defined and has no
        appropriate neighbours"""
        block_0 = self.make_block(0)
        self.assertFalse(block_0.copy_grading())

    def test_copy_grading_neighbour_undefined(self):
        """block.copy_grading from an undefined neighbour"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        # clear chops from block_1
        block_1.axes[0].wires.chops = []
        block_1.axes[1].wires.chops = []
        block_1.axes[2].wires.chops = []

        block_0.add_neighbour(block_1)

        self.assertFalse(block_0.copy_grading())

    def test_copy_grading_neighbour_defined(self):
        """block.copy_grading from a defined neighbour"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.grade()
        block_1.grade()

        block_0.add_neighbour(block_1)

        self.assertTrue(block_0.copy_grading())

    def test_output(self):
        """Output of a simple block"""
        # just to make sure it works
        block_0 = self.make_block(0)
        block_0.axes[0].wires.chops = []

        block_0.chop(0, Chop(count=10))
        block_0.chop(1, Chop(count=10))
        block_0.chop(2, Chop(count=10))

        block_0.cell_zone = "test_zone"
        block_0.comment = "test_comment"

        for axis in block_0.axes:
            axis.wires.grade()

        expected_description = (
            "\thex ( 0 1 2 3 4 5 6 7 ) test_zone ( 10 10 10 )" " simpleGrading ( 1 1 1 ) // 0 test_comment\n"
        )

        self.assertEqual(block_0.description, expected_description)


class BlockEdgeGradingTests(BlockTestCase):
    """Refer to test_edge_grading.py for more involved (function) tests"""

    def make_vertices(self, index: int) -> List[Vertex]:
        vertices = super().make_vertices(index)

        if index == 1:
            # move vertices 8 and 10 away so that an irregular block shape is
            # obtained, appropriate for edgeGrading in axes 0 and 1
            displacement = f.vector(0.3, -0.3, 0)

            vertices[1].translate(displacement)

        return vertices

    def make_block(self, index: int) -> Block:
        block = super().make_block(index)

        # clear chops so that they are added manually for each test
        for axis in block.axes:
            axis.wires.chops = []

        return block

    def chop(self, block: Block, axis, **kwargs):
        block.chop(axis, Chop(**kwargs))

    def calculate(self, block):
        block.grade()

    def test_edge_grading_output(self):
        """Switch to edgeGrading when a chop is to be 'preserved'"""
        block = self.make_block(1)
        self.chop(block, 0, start_size=0.01, end_size=0.2, preserve="start_size")
        self.chop(block, 1, count=10)
        self.chop(block, 2, count=10)
        self.calculate(block)

        self.assertTrue("edgeGrading" in block.format_grading())

    def test_edge_multigrading(self):
        """Switch to edgeGrading when `preserve` keyword is provided"""
        block = self.make_block(1)
        self.chop(block, 0, length_ratio=0.5, start_size=0.01, end_size=0.1)
        self.chop(block, 0, length_ratio=0.5, end_size=0.01, start_size=0.1, preserve="end_size")
        self.chop(block, 1, count=10)
        self.chop(block, 2, count=10)
        self.calculate(block)

        self.assertTrue("edgeGrading" in block.format_grading())

    def test_copy_grading(self):
        """Copy data from neighbours"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        self.chop(block_0, 0, count=10)
        self.chop(block_0, 2, count=10)
        self.chop(block_1, 0, start_size=0.1, end_size=0.01, preserve="end_size")
        self.chop(block_1, 1, count=10)

        self.calculate(block_0)
        self.calculate(block_1)

        block_0.add_neighbour(block_1)
        block_1.add_neighbour(block_0)

        block_0.copy_grading()
        block_1.copy_grading()

        # block_0 must be simpleGraded, block_1 edge
        self.assertTrue("simpleGrading" in block_0.format_grading())
        self.assertTrue("edgeGrading" in block_1.format_grading())

    def test_single_grading(self):
        """Output simpleGrading when there is no need for edgeGrading"""
        block = self.make_block(2)
        self.chop(block, 0, count=10)
        self.chop(block, 1, count=10)
        self.chop(block, 2, start_size=0.1, end_size=0.01, preserve="start_size")

        self.calculate(block)

        # block_0 must be simpleGraded, block_1 edge
        self.assertTrue("simpleGrading" in block.format_grading())


class WireframeTests(BlockTestCase):
    """Workings of the Frame object in a Block"""

    def setUp(self):
        self.vertices = self.make_vertices(0)
        self.block = self.make_block(0)

    def test_wires_count(self):
        """Each corner has exactly 3 couples"""
        for wires in self.block.wires:
            self.assertEqual(len(wires), 3)

    def test_wire_indexes(self):
        """Check that indexes are generated exactly as the sketch dictates"""
        # hand-typed
        expected_indexes = (
            (1, 3, 4),  # 0
            (0, 2, 5),  # 1
            (1, 3, 6),  # 2
            (0, 2, 7),  # 3
            (0, 5, 7),  # 4
            (1, 4, 6),  # 5
            (2, 5, 7),  # 6
            (3, 4, 6),  # 7
        )

        for i, wires in enumerate(self.block.wires):
            generated = set(wires.keys())
            expected = set(expected_indexes[i])

            self.assertSetEqual(generated, expected)

    def test_axis_count(self):
        """Check that each axis has exactly 4 wires"""
        for axis in range(3):
            self.assertEqual(len(self.block.axes[axis].wires.wires), 4)

    @parameterized.expand(((0, 2), (1, 3), (0, 6), (1, 7)))
    def test_find_wire_fail(self, corner_1, corner_2):
        """There are no wires in face or volume diagonals"""
        with self.assertRaises(KeyError):
            _ = self.block.wires[corner_1][corner_2]

    @parameterized.expand(((0, 1), (1, 0), (0, 4), (4, 0)))
    def test_find_wire_success(self, corner_1, corner_2):
        """Find wire by __getitem__"""
        self.assertEqual(type(self.block.wires[corner_1][corner_2]), Wire)

    @parameterized.expand(((0,), (1,), (2,)))
    def test_get_axis_wires_length(self, axis):
        """Number of wires for each axis"""
        self.assertEqual(len(self.block.get_axis_wires(axis)), 4)

    @parameterized.expand(((0, 2), (1, 2), (2, 0)))
    def test_edge_list_empty(self, block_index, edge_count):
        """A block with only line edges must have an empty edge_list"""
        block = self.make_block(block_index)
        self.assertEqual(len(block.edge_list), edge_count)

    def test_index_exception(self):
        """Raise an exception when wrong indexes are provided to add_edge()"""
        block = self.make_block(0)

        with self.assertRaises(ValueError):
            block.add_edge(0, 9, Arc([1, 1, 1]))
