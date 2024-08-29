import unittest

import numpy as np

from classy_blocks.base.exceptions import InconsistentGradingsError
from classy_blocks.construct.edges import Arc
from classy_blocks.construct.operations.box import Box
from classy_blocks.construct.operations.extrude import Extrude
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.mesh import Mesh


class EdgeGradingExampleTests(unittest.TestCase):
    def setUp(self):
        """An example case, but thoroughly tested"""
        mesh = Mesh()

        start = Box([0, 0, 0], [1, 1, 0.1])
        start.chop(0, start_size=0.1)
        start.chop(1, length_ratio=0.5, start_size=0.01, c2c_expansion=1.2, preserve="start_size")
        start.chop(1, length_ratio=0.5, end_size=0.01, c2c_expansion=1 / 1.2, preserve="end_size")
        start.chop(2, count=1)
        mesh.add(start)

        expand_start = start.get_face("right")
        expand = Loft(expand_start, expand_start.copy().translate([1, 0, 0]).scale(2))
        expand.chop(2, start_size=0.1)
        mesh.add(expand)

        contract_start = expand.get_face("top")
        contract = Loft(contract_start, contract_start.copy().translate([1, 0, 0]).scale(0.25))
        contract.chop(2, start_size=0.1)
        mesh.add(contract)

        # rotate the end block to test grading on non-aligned blocks
        end = Extrude(contract.get_face("top"), 1)
        end.rotate(np.pi, [0, 0, 1])
        end.chop(2, start_size=0.1)

        mesh.add(end)

        mesh.assemble()
        mesh.grade()

        self.mesh = mesh

    def test_axis_simple_grading(self):
        # blocks 0 and 3 are simpleGrading because their edges are equal in each axis

        for i in (0, 3):
            for axis in self.mesh.blocks[i].axes:
                self.assertTrue(axis.is_simple)

    def test_axis_edge_grading(self):
        # blocks 1 and 2 are edgeGraded in axis 1
        for i in (1, 2):
            self.assertFalse(self.mesh.blocks[i].axes[1].is_simple)

    def test_block_grading_simple(self):
        # blocks 0 and 3 can be simpleGraded
        for i in (0, 3):
            self.assertTrue("simpleGrading" in self.mesh.blocks[i].description)

    def test_block_grading_edge(self):
        # blocks 1 and 2 must be edge graded
        for i in (1, 2):
            self.assertTrue("edgeGrading" in self.mesh.blocks[i].description)


class EdgeGradingTests(unittest.TestCase):
    """Border-cases tests and whatnot"""

    def setUp(self):
        self.mesh = Mesh()

    def prepare(self):
        self.mesh.assemble()
        self.mesh.grade()

    def test_inconsistent_wires(self):
        box_left = Box([0, 0, 0], [1, 1, 1])
        box_right = box_left.copy().translate([2, 0, 0])
        box_mid = box_left.copy().translate([1, 0, 0])

        for box in (box_left, box_right, box_mid):
            box.chop(0, count=10)
            self.mesh.add(box)

        box_left.chop(1, count=10)

        # chop left and right unevenly
        box_left.chop(2, count=5)
        box_right.chop(2, count=15)

        with self.assertRaises(InconsistentGradingsError):
            self.prepare()

    def test_curved_block(self):
        """Edge-grade a perfect cube but with a curved edge"""
        box = Box([0, 0, 0], [1, 1, 1])
        box.add_side_edge(0, Arc([-0.5, -0.5, 0.5]))

        box.chop(0, count=10)
        box.chop(1, count=10)
        box.chop(2, start_size=0.1, end_size=0.01, preserve="end_size")

        self.mesh.add(box)
        self.prepare()

        self.assertFalse(self.mesh.blocks[0].axes[2].is_simple)

    def test_no_preserve(self):
        """Do not edge grade anything without a 'preserve' keyword"""
        box = Box([0, 0, 0], [1, 1, 1])
        box.add_side_edge(0, Arc([-0.5, -0.5, 0.5]))

        box.chop(0, count=10)
        box.chop(1, count=10)
        box.chop(2, start_size=0.1, end_size=0.01)

        self.mesh.add(box)
        self.prepare()

        self.assertTrue(self.mesh.blocks[0].axes[2].is_simple)
