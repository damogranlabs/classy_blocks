from parameterized import parameterized
from tests.fixtures.block import BlockTestCase

from classy_blocks.items.wire import Wire

class BlockTests(BlockTestCase):
    """Block item tests"""
    @parameterized.expand((
        ((1, 2), (0, 3)), # corner_1, corner_2 of block_0 and corner_1, corner_2 of block_1
        ((5, 6), (4, 7)),
        ((1, 5), (0, 4)),
        ((2, 6), (3, 7)),
    ))
    def test_add_neighbour_1_wires(self, this_corners, nei_corners):
        """Two blocks that share a 'side' a.k.a. face"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.add_neighbour(block_1)

        # these two blocks share the whole face (1 2 6 5)
        # 4 wires altogether
        self.assertEqual(
            block_0.wires[this_corners[0]][this_corners[1]].coincidents,
            {block_1.wires[nei_corners[0]][nei_corners[1]]}
        )

    def test_add_neighbour_1_axes(self):
        """Two blocks that share a 'side' a.k.a. face"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.add_neighbour(block_1)

        # block_1 is block_0's neighbour in axes 1 and 2
        self.assertListEqual(block_0.axes[0].neighbours, [])
        self.assertListEqual(block_0.axes[1].neighbours, [block_1.axes[1]])
        self.assertListEqual(block_0.axes[2].neighbours, [block_1.axes[2]])

    def test_add_neighbour_2_wires(self):
        """Two blocks that share an edge only"""
        block_0 = self.make_block(0)
        block_2 = self.make_block(2)

        block_0.add_neighbour(block_2)

        # there must be only 1 wire that only has 1 neighbour
        self.assertEqual(
            block_0.wires[2][6].coincidents,
            {block_2.wires[0][4]}
        )

    def test_add_neighbour_2_axes(self):
        """Two blocks that share an edge only"""
        block_0 = self.make_block(0)
        block_2 = self.make_block(2)

        block_0.add_neighbour(block_2)
        # block_2 is block_0'2 neighbour only on axis 2
        self.assertListEqual(block_0.axes[0].neighbours, [])
        self.assertListEqual(block_0.axes[1].neighbours, [])
        self.assertListEqual(block_0.axes[2].neighbours, [block_2.axes[2]])


    def test_add_neighbour_3(self):
        """Where three blocks meet, there's a wire with 2 coincident wires"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)
        block_2 = self.make_block(2)

        block_0.add_neighbour(block_1)
        block_0.add_neighbour(block_2)

        self.assertEqual(
            block_0.wires[2][6].coincidents,
            {block_1.wires[3][7], block_2.wires[0][4]}
        )
    
    def test_add_neighbour_twice(self):
        """Add the same neighbour twice"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.add_neighbour(block_1)
        block_0.add_neighbour(block_1)

        self.assertEqual(len(block_0.axes[1].neighbours), 1)
    
    def test_add_self(self):
        """Add the same block as a neighbour"""
        block_0 = self.make_block(0)

        block_0.add_neighbour(block_0)

        self.assertEqual(len(block_0.axes[0].neighbours), 0)
    
    def test_add_neighbour_inverted(self):
        """Add a neighbour with an inverted axis"""
        # TODO
        a = 1/0
    
    def test_add_foreign(self):
        """Try to add a block that is not in contact
        as a neighbour"""
        # TODO
        a = 1/0
    
    def test_project_face_side(self):
        """Project a face and check if appropriate Side contains geometry"""
        block = self.make_block(0)

        block.project_face('bottom', 'terrain', edges=False)

        self.assertListEqual(block.sides['bottom'].project_to, ['terrain'])
    
    def test_project_face_edges(self):
        """Project a face and check if there are projected edges in edge_list"""
        block = self.make_block(0)

        block.project_face('bottom', 'terrain', edges=True)

        n_project_edges = 0
        for edge in block.edge_list:
            if edge.kind == 'project':
                n_project_edges += 1
            
        self.assertEqual(n_project_edges, 4)

    def test_set_patch_single(self):
        """Set a patch on a single side"""
        block = self.make_block(0)
        block.set_patch('bottom', 'terrain')

        self.assertEqual(block.sides['bottom'].patch_name, 'terrain')
    
    def test_set_patch_multiple(self):
        """Set a patch to multiple sides at once"""
        block = self.make_block(0)

        sides = ['front', 'left', 'right']

        block.set_patch(sides, 'walls')

        for side in sides:
            self.assertEqual(block.sides[side].patch_name, 'walls')

class WireframeTests(BlockTestCase):
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
            (1, 3, 4), # 0
            (0, 2, 5), # 1
            (1, 3, 6), # 2
            (0, 2, 7), # 3
            (0, 5, 7), # 4
            (1, 4, 6), # 5
            (2, 5, 7), # 6
            (3, 4, 6), # 7
        )

        for i, wires in enumerate(self.block.wires):
            generated = set(wires.keys())
            expected = set(expected_indexes[i])

            self.assertSetEqual(generated, expected)

    def test_axis_count(self):
        """Check that each axis has exactly 4 wires"""
        for axis in range(3):
            self.assertEqual(len(self.block.axes[axis].wires), 4)

    @parameterized.expand(((0, 2), (1, 3), (0, 6), (1, 7)))
    def test_find_wire_fail(self, corner_1, corner_2):
        """There are no wires in face or volume diagonals"""
        with self.assertRaises(KeyError):
            _ = self.block.wires[corner_1][corner_2]

    @parameterized.expand(((0, 1), (1, 0), (0, 4), (4, 0)))
    def test_find_wire_success(self, corner_1, corner_2):
        """Find wire by __getitem__"""
        self.assertEqual(type(self.block.wires[corner_1][corner_2]), Wire)
