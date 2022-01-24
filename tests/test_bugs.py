import unittest

from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.shapes import ExtrudedRing

from tests.fixtures import ExecutedTestsBase

class TestBugs(ExecutedTestsBase):
    def test_block_orientation(self):
        """ block.chop() calculates a different count when blocks are
        oriented (rotated) differently """
        d_shaft = 12e-3
        d_2 = 63e-3
        h = 20e-3

        cell_size = 1.5e-3

        self.mesh = Mesh()

        inner_ring = ExtrudedRing(
            [0, 0, 0],
            [0, 0, h],
            [d_shaft/2, 0, 0],
            d_2/2
        )

        inner_ring.chop_axial(start_size=cell_size)
        inner_ring.chop_radial(start_size=cell_size)
        inner_ring.chop_tangential(start_size=cell_size)

        self.mesh.add(inner_ring)
        self.run_and_check()