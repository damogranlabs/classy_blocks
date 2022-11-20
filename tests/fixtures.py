import unittest

from classy_blocks.define.edge import Edge
from classy_blocks.define.block import Block
from classy_blocks.process.mesh import Mesh

write_mesh = False

class FixturedTestCase(unittest.TestCase):
    """common setUp for block and mesh tests"""
    def setUp(self):
        """Create blocks but don't add them to the mesh yet"""
        # a test mesh:
        # 3 blocks, extruded in z-direction
        # (these indexes refer to 'fl' and 'cl' variables,
        # not vertex.mesh_index)
        #
        #  ^ y-axis
        #  |
        #  |   7---6
        #      | 2 |
        #  3---2---5
        #  | 0 | 1 |
        #  0---1---4   ---> x-axis
        #
        # After adding the blocks, the following mesh indexes are
        # in mesh.vertices:
        # 
        # Bottom 'floor':
        #
        #  ^ y-axis
        #  |
        #  |  13--12
        #      | 2 |
        #  3---2---9
        #  | 0 | 1 |
        #  0---1---8   ---> x-axis
        #
        # Top 'floor':
        #
        #  ^ y-axis
        #  |
        #  |  15--14
        #      | 2 |
        #  7---6--11
        #  | 0 | 1 |
        #  4---5--10   ---> x-axis

        fl = [  # points on the 'floor'; z=0
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [2, 0, 0],  # 4
            [2, 1, 0],  # 5
            [2, 2, 0],  # 6
            [1, 2, 0],  # 7
        ]

        cl = [[p[0], p[1], 1] for p in fl]  # points on ceiling; z = 1

        # create blocks from the most basic way;
        # block.create_from_points()
        self.block_0_points = [
            fl[0],
            fl[1],
            fl[2],
            fl[3],
            cl[0],
            cl[1],
            cl[2],
            cl[3],
        ]

        self.block_1_points = [
            fl[1],
            fl[4],
            fl[5],
            fl[2],
            cl[1],
            cl[4],
            cl[5],
            cl[2],
        ]

        self.block_2_points = [
            fl[2],
            fl[5],
            fl[6],
            fl[7],
            cl[2],
            cl[5],
            cl[6],
            cl[7],
        ]

        # the most low-level way of creating a block is from 'raw' points
        self.block_0 = Block.create_from_points(self.block_0_points)

        # block 0 has curved edges of all types
        self.block_0.add_edge(0, 1, [0.5, -0.25, 0]) # arc edge
        self.block_0.add_edge(1, 2, [[1.1, 0.25, 0], [1.05, 0.5, 0], [1.1, 0.75, 0]])  # spline edge

        self.block_0.chop(0, count=6)

        self.block_1 = Block.create_from_points(self.block_1_points)
        # additional edge in block 2 that must not be included (already in block_1_edges)
        self.block_1.add_edge(3, 0, [0.5, -0.1, 1])
        self.block_1.add_edge(0, 1, [1.5, 0, 0])  # collinear point; invalid edge must be dropped

        self.block_1.chop(0, count=5)
        self.block_1.chop(1, count=6)

        self.block_2 = Block.create_from_points(self.block_2_points)
        self.block_2.chop(1, count=8)
        self.block_2.chop(2, count=7)

        # other block data
        self.block_0.description = "Test"
        self.block_0.set_patch("left", "inlet")
        self.block_0.set_patch(["bottom", "top", "front", "back"], "walls")

        self.block_1.set_patch(["bottom", "top", "right", "front"], "walls")

        self.block_2.set_patch("back", "outlet")
        self.block_2.set_patch(["bottom", "top", "left", "right"], "walls")

        self.mesh = Mesh()

    def add(self):
        """Only add blocks to the mesh, no additional processing"""
        self.mesh.add(self.block_0)
        self.mesh.add(self.block_1)
        self.mesh.add(self.block_2)
        
    def prepare(self):
        """Prepare the mesh for writing"""
        self.add()
        self.mesh.prepare()

if __name__ == "__main__":
    unittest.main()
