import numpy as np

from classy_blocks.construct.operations.box import Box
from classy_blocks.construct.shapes.cylinder import Cylinder
from classy_blocks.construct.shapes.sphere import EighthSphere
from classy_blocks.mesh import Mesh
from tests.fixtures.block import BlockTestCase


class MeshTests(BlockTestCase):
    def setUp(self):
        self.mesh = Mesh()

    def test_is_not_assembled(self):
        """A fresh mesh: not assembled"""
        self.assertFalse(self.mesh.is_assembled)

    def test_is_assembled(self):
        """An assembled mesh"""
        # If any processing has been done
        self.mesh.add(self.make_loft(0))
        self.mesh.assemble()

        self.assertTrue(self.mesh.is_assembled)

    def test_assemble(self):
        """Add a couple of operations and run assemble"""
        for i in range(3):
            self.mesh.add(self.make_loft(i))

        self.mesh.assemble()

        self.assertEqual(len(self.mesh.blocks), 3)

    def test_merged_multi(self):
        """Face merge multiple touching blocks"""
        # a 2x2 array of boxes
        center = [0.0, 0.0, 0.0]

        # |----|----|
        # | 01 | 00 |
        # |----C----|
        # | 11 | 10 |
        # |----|----|

        box_00 = Box(center, [1, 1, 1])
        box_01 = Box(center, [-1, 1, 1])
        box_11 = Box(center, [-1, -1, 1])
        box_10 = Box(center, [1, -1, 1])

        box_00.set_patch("left", "left_00")
        box_00.set_patch("front", "front_00")

        box_01.set_patch("right", "right_01")
        box_01.set_patch("front", "front_01")

        box_11.set_patch("right", "right_11")
        box_11.set_patch("back", "back_11")

        box_10.set_patch("left", "left_10")
        box_10.set_patch("back", "back_10")

        self.mesh.add(box_00)
        self.mesh.add(box_01)
        self.mesh.add(box_11)
        self.mesh.add(box_10)

        self.mesh.merge_patches("left_00", "right_01")
        self.mesh.merge_patches("front_00", "back_10")
        self.mesh.merge_patches("front_01", "back_11")
        self.mesh.merge_patches("left_10", "right_11")

        self.mesh.assemble()

        # all vertices must be duplicated
        self.assertEqual(len(self.mesh.vertices), 32)

    def test_cell_zone_operation(self):
        """Assign cell zone from an operation"""
        box = Box([0, 0, 0], [1, 1, 1])
        box.set_cell_zone("mrf")

        self.mesh.add(box)
        self.mesh.assemble()

        for block in self.mesh.blocks:
            self.assertEqual(block.cell_zone, "mrf")

    def test_cell_zone_shape(self):
        """Assign cell zone from an operation"""
        cyl = Cylinder([0, 0, 0], [1, 0, 0], [0, 1, 0])
        cyl.set_cell_zone("mrf")

        self.mesh.add(cyl)
        self.mesh.assemble()

        for block in self.mesh.blocks:
            self.assertEqual(block.cell_zone, "mrf")

    def test_chop_shape_axial(self):
        """Axial chop of a shape"""
        cyl = Cylinder([0, 0, 0], [1, 0, 0], [0, 1, 0])
        cyl.chop_axial(count=10)
        cyl.chop_radial(count=1)
        cyl.chop_tangential(count=1)

        self.mesh.add(cyl)
        self.mesh.assemble().finalize()

        for block in self.mesh.blocks:
            self.assertEqual(block.axes[2].count, 10)

    def test_chop_cylinder_tangential(self):
        """Cylinder chops differently from rings"""
        cyl = Cylinder([0, 0, 0], [1, 0, 0], [0, 1, 0])
        cyl.chop_tangential(count=10)
        cyl.chop_radial(count=1)
        cyl.chop_axial(count=1)

        self.mesh.add(cyl)
        self.mesh.assemble().finalize()

        for block in self.mesh.blocks:
            self.assertEqual(block.axes[1].count, 10)

    def test_set_default_patch(self):
        self.mesh.set_default_patch("terrain", "wall")

        self.assertDictEqual(self.mesh.settings.default_patch, {"name": "terrain", "kind": "wall"})

    def test_modify_patch(self):
        self.mesh.modify_patch("terrain", "wall", ["transform none"])

        self.assertEqual(self.mesh.settings.patch_settings["terrain"], ["wall", "transform none"])

    def test_operations(self):
        """Add an op and a shape and check operation count"""
        # a single block
        box = Box([0, 0, 0], [1, 1, 1])
        self.mesh.add(box)

        # 12 blocks
        cylinder = Cylinder([2, 0, 0], [3, 0, 0], [2, 1, 0])
        self.mesh.add(cylinder)

        self.assertEqual(len(self.mesh.operations), 13)

    def test_with_geometry(self):
        esph = EighthSphere([0, 0, 0], [1, 0, 0], [0, 0, 1])
        esph.chop_axial(count=10)
        esph.chop_radial(count=10)
        esph.chop_tangential(count=10)

        self.mesh.add(esph)
        self.mesh.assemble()

        self.assertEqual(len(self.mesh.settings.geometry), 1)

    def test_backport(self):
        box = Box([0, 0, 0], [1, 1, 1])
        self.mesh.add(box)
        self.mesh.assemble()
        self.mesh.vertices[0].move_to([-1, -1, -1])

        self.mesh.backport()

        np.testing.assert_array_equal(box.point_array[0], [-1, -1, -1])

    def test_backport_empty(self):
        self.mesh.add(self.make_loft(0))

        with self.assertRaises(RuntimeError):
            self.mesh.backport()

    def test_delete(self):
        boxes = [
            Box([0, 0, 0], [1, 1, 1]),
            Box([1, 0, 0], [2, 1, 1]),
            Box([0, 1, 0], [1, 2, 1]),
            Box([1, 1, 0], [2, 2, 1]),
        ]

        for box in boxes:
            self.mesh.add(box)

        self.mesh.delete(boxes[0])

        self.mesh.assemble()

        self.assertFalse(self.mesh.blocks[0].visible)

    def test_assemble_noncoincident(self):
        """Assemble a mesh with non-coincident vertices"""
        box = Box([0, 0, 0], [1, 1, 1])
        box.set_patch("left", "left_patch")
        box.set_patch("front", "front_patch")
        self.mesh.add(box)

        box2 = Box([1.001, 1.001, 1.001], [2, 2, 2])
        box2.set_patch("right", "right_patch")
        box2.set_patch("back", "back_patch")
        self.mesh.add(box2)

        self.mesh.assemble(merge_tol=0.002)

        self.assertEqual(len(self.mesh.vertices), 15)
