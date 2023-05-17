from tests.fixtures.block import BlockTestCase

from classy_blocks.mesh import Mesh
from classy_blocks.construct.operations.box import Box
from classy_blocks.construct.shapes.cylinder import Cylinder
from classy_blocks.construct.shapes.rings import ExtrudedRing


class MeshTests(BlockTestCase):
    def setUp(self):
        self.mesh = Mesh()

    def test_settings_output(self):
        """Proper formatting of settings"""
        self.mesh.settings["prescale"] = 1
        self.mesh.settings["scale"] = 0.001
        self.mesh.settings["mergeType"] = "points"

        expected = "prescale 1;\nscale 0.001;\nmergeType points;\n\n"

        self.assertEqual(self.mesh.format_settings(), expected)

    def test_add_vertices(self):
        """Create Vertices from Operation"""
        # re-use the same vertices
        loft = self.make_loft(0)
        self.mesh._add_vertices(loft)

        another_loft = self.make_loft(1)
        self.mesh._add_vertices(another_loft)

        self.assertEqual(len(self.mesh.vertex_list.vertices), 12)

    def test_add_vertices_slave(self):
        """Add two lofts where sides are face-merged"""
        # this time, no reusing of vertices is allowed
        loft_left = self.make_loft(0)
        loft_left.set_patch("right", "master")

        loft_right = self.make_loft(1)
        loft_right.set_patch("left", "slave")

        self.mesh.merge_patches("master", "slave")

        self.mesh._add_vertices(loft_left)
        self.mesh._add_vertices(loft_right)

        self.assertEqual(len(self.mesh.vertex_list.vertices), 16)

    def test_add_vertices_slave_flipped(self):
        """Add an operation with a 'slave' patch first"""
        # it should make no difference
        loft_left = self.make_loft(0)
        loft_left.set_patch("right", "slave")

        loft_right = self.make_loft(1)
        loft_right.set_patch("left", "master")

        self.mesh.merge_patches("master", "slave")

        self.mesh._add_vertices(loft_left)
        self.mesh._add_vertices(loft_right)

        self.assertEqual(len(self.mesh.vertex_list.vertices), 16)

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

        self.assertEqual(len(self.mesh.block_list.blocks), 3)

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

        self.mesh._add_vertices(box_00)
        self.mesh._add_vertices(box_01)
        self.mesh._add_vertices(box_11)
        self.mesh._add_vertices(box_10)

        # all vertices must be duplicated
        self.assertEqual(len(self.mesh.vertex_list.vertices), 32)

    def test_cell_zone_operation(self):
        """Assign cell zone from an operation"""
        box = Box([0, 0, 0], [1, 1, 1])
        box.set_cell_zone('mrf')

        self.mesh.add(box)
        self.mesh.assemble()

        for block in self.mesh.block_list.blocks:
            self.assertEqual(block.cell_zone, 'mrf')

    def test_cell_zone_shape(self):
        """Assign cell zone from an operation"""
        cyl = Cylinder([0, 0, 0], [1, 0, 0], [0, 1, 0])
        cyl.set_cell_zone('mrf')

        self.mesh.add(cyl)
        self.mesh.assemble()

        for block in self.mesh.block_list.blocks:
            self.assertEqual(block.cell_zone, 'mrf')
    
    def test_chop_shape_axial(self):
        """Axial chop of a shape"""
        cyl = Cylinder([0, 0, 0], [1, 0, 0], [0, 1, 0])
        cyl.chop_axial(count=10)
        cyl.chop_radial(count=1)
        cyl.chop_tangential(count=1)

        self.mesh.add(cyl)
        self.mesh.assemble()
        self.mesh.block_list.propagate_gradings()
        
        for block in self.mesh.block_list.blocks:
            self.assertEqual(block.axes[2].grading.count, 10)
    
    def test_chop_cylinder_tangential(self):
        """Cylinder chops differently from rings"""
        cyl = Cylinder([0, 0, 0], [1, 0, 0], [0, 1, 0])
        cyl.chop_tangential(count=10)
        cyl.chop_radial(count=1)
        cyl.chop_axial(count=1)

        self.mesh.add(cyl)
        self.mesh.assemble()
        self.mesh.block_list.propagate_gradings()
        
        for block in self.mesh.block_list.blocks:
            self.assertEqual(block.axes[1].grading.count, 10)

