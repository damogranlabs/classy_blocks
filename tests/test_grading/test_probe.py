from typing import Set, get_args

import numpy as np
from parameterized import parameterized

from classy_blocks.grading.autograding.catalogue import get_block_from_axis
from classy_blocks.grading.autograding.probe import Probe, get_defined_wall_vertices
from classy_blocks.items.vertex import Vertex
from classy_blocks.mesh import Mesh
from classy_blocks.modify.find.shape import RoundSolidFinder
from classy_blocks.types import DirectionType
from tests.test_grading.test_autograde import AutogradeTestsBase


class ProbeTests(AutogradeTestsBase):
    def test_block_from_axis_fail(self):
        mesh_1 = self.mesh
        mesh_1.add(self.get_stack())
        mesh_1.assemble()

        mesh_2 = Mesh()
        mesh_2.add(self.get_stack())
        mesh_2.assemble()

        with self.assertRaises(RuntimeError):
            get_block_from_axis(mesh_1, mesh_2.blocks[0].axes[0])

    @parameterized.expand((("min", 0.196889), ("max", 0.8), ("avg", 0.470923)))
    def test_get_row_length(self, take, length):
        self.mesh.add(self.get_frustum())
        self.mesh.assemble()

        probe = Probe(self.mesh)
        row = probe.get_rows(0)[0]

        self.assertAlmostEqual(row.get_length(take), length, places=4)

    @parameterized.expand(((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 2)))
    def test_get_blocks_on_layer(self, block, axis):
        self.mesh.add(self.get_stack())
        self.mesh.assemble()

        probe = Probe(self.mesh)
        blocks = probe.get_row_blocks(self.mesh.blocks[block], axis)

        self.assertEqual(len(blocks), 16)

    @parameterized.expand(((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,)))
    def test_block_from_axis(self, index):
        self.mesh.add(self.get_cylinder())
        self.mesh.assemble()

        for axis in get_args(DirectionType):
            block = self.mesh.blocks[index]

            self.assertEqual(block, get_block_from_axis(self.mesh, block.axes[axis]))

    @parameterized.expand(((0,), (1,), (2,)))
    def test_get_layers(self, axis):
        self.mesh.add(self.get_stack())
        self.mesh.assemble()

        probe = Probe(self.mesh)
        layers = probe.get_rows(axis)

        self.assertEqual(len(layers), 4)

    @parameterized.expand(
        (
            # axis, layer, block indexes
            (0, 0, {5, 0, 3, 10}),
            (0, 1, {6, 1, 2, 9}),
            (0, 2, {4, 5, 6, 7, 8, 9, 10, 11}),
            (1, 0, {7, 1, 0, 4}),
            (1, 1, {8, 2, 3, 11}),
            (2, 0, set(range(12))),
        )
    )
    def test_get_blocks_cylinder(self, axis, row, blocks):
        self.mesh.add(self.get_cylinder())
        self.mesh.assemble()

        probe = Probe(self.mesh)
        indexes = set()

        for block in probe.catalogue.rows[axis][row].blocks:
            indexes.add(block.index)

        self.assertSetEqual(indexes, blocks)

    @parameterized.expand(
        (
            # axis, layer, block indexes
            (1, 0, {0, 1, 2, 3}),
            (1, 1, {4, 5, 6, 7}),
            (1, 2, {8, 9, 10, 11}),
            (1, 3, {12, 13, 14, 15}),
        )
    )
    def test_get_blocks_inverted(self, axis, row, blocks):
        shape = self.get_flipped_stack().shapes[1]
        self.mesh.add(shape)
        self.mesh.assemble()

        probe = Probe(self.mesh)
        indexes = set()

        for block in probe.catalogue.rows[axis][row].blocks:
            indexes.add(block.index)

        self.assertSetEqual(indexes, blocks)

    def test_wall_vertices_defined(self) -> None:
        """Catch wall vertices from explicitly defined wall patches"""
        cylinder = self.get_cylinder()
        cylinder.set_outer_patch("outer")

        self.mesh.add(cylinder)
        self.mesh.modify_patch("outer", "wall")
        self.mesh.assemble()

        finder = RoundSolidFinder(self.mesh, cylinder)
        shell_vertices = finder.find_shell(True).union(finder.find_shell(False))
        wall_vertices: Set[Vertex] = set()

        wall_vertices.update(get_defined_wall_vertices(self.mesh))

        self.assertSetEqual(shell_vertices, wall_vertices)

    def test_wall_vertices_default(self) -> None:
        """Catch wall vertices from default patch"""
        cylinder = self.get_cylinder()
        cylinder.set_start_patch("inlet")
        cylinder.set_end_patch("outlet")
        self.mesh.add(cylinder)

        self.mesh.set_default_patch("outer", "wall")
        self.mesh.assemble()

        probe = Probe(self.mesh)

        finder = RoundSolidFinder(self.mesh, cylinder)
        shell_vertices = finder.find_shell(True).union(finder.find_shell(False))
        wall_vertices: Set[Vertex] = set()

        for block in self.mesh.blocks:
            wall_vertices.update(probe.get_wall_vertices(block))

        self.assertSetEqual(shell_vertices, wall_vertices)

    def test_wall_vertices_combined(self) -> None:
        cylinder = self.get_cylinder()
        cylinder.set_end_patch("outlet")

        cylinder.set_start_patch("bottom")
        self.mesh.add(cylinder)

        self.mesh.modify_patch("bottom", "wall")

        self.mesh.set_default_patch("outer", "wall")
        self.mesh.assemble()

        probe = Probe(self.mesh)

        finder = RoundSolidFinder(self.mesh, cylinder)
        shell_vertices = finder.find_shell(True).union(finder.find_shell(False)).union(finder.find_core(False))
        wall_vertices: Set[Vertex] = set()

        for block in self.mesh.blocks:
            wall_vertices.update(probe.get_wall_vertices(block))

        self.assertSetEqual(shell_vertices, wall_vertices)

    def test_flipped_simple(self):
        shape = self.get_stack().shapes[0]
        shape.grid[0][1].rotate(np.pi, [0, 0, 1])

        self.mesh.add(shape)
        self.mesh.assemble()

        probe = Probe(self.mesh)
        row = probe.get_rows(1)[0]

        self.assertListEqual([entry.flipped for entry in row.entries], [False, True, False, False])

    @parameterized.expand(
        (
            ((1,), 0, 1),
            ((1, 2), 0, 1),
            ((1, 2), 0, 2),
            ((1, 2, 5), 1, 1),
            ((0,), 0, 1),
            ((0,), 0, 2),
        )
    )
    def test_flipped_shape(self, flip_indexes, check_row, check_index):
        stack = self.get_stack().shapes[0]

        for i in flip_indexes:
            stack.operations[i].rotate(np.pi, [0, 0, 1])

        self.mesh.add(stack)
        self.mesh.assemble()

        probe = Probe(self.mesh)

        self.assertTrue(probe.get_rows(1)[check_row].entries[check_index].flipped)

    @parameterized.expand(
        (
            (0, 4, True),
            (1, 5, True),
            (2, 6, True),
            (3, 7, True),
            (0, 1, True),
            (3, 2, True),
            (4, 5, True),
            (7, 6, True),
            (0, 3, False),
            (1, 2, False),
            (4, 7, False),
            (5, 6, False),
        )
    )
    def test_wire_boundaries_explicit(self, wire_start, wire_end, starts_at_wall):
        box = self.get_box()
        box.set_patch(["bottom", "left", "right"], "wallPatch")
        self.mesh.add(box)

        self.mesh.modify_patch("wallPatch", "wall")
        self.mesh.assemble()

        probe = Probe(self.mesh)
        block = self.mesh.blocks[0]
        info = probe.get_wire_info(block.wires[wire_start][wire_end], block)

        self.assertEqual(info.starts_at_wall, starts_at_wall)

    @parameterized.expand(
        (
            (0, 4, True),
            (1, 5, True),
            (2, 6, True),
            (3, 7, True),
            (0, 1, True),
            (3, 2, True),
            (4, 5, True),
            (7, 6, True),
            (0, 3, False),
            (1, 2, False),
            (4, 7, False),
            (5, 6, False),
        )
    )
    def test_wire_boundaries_default(self, wire_start, wire_end, starts_at_wall):
        # same situation as the 'explicit' test but 'patch' patch types are defined
        # and walls are the default
        box = self.get_box()
        box.set_patch(["top", "front", "back"], "patchPatch")
        self.mesh.add(box)

        self.mesh.set_default_patch("defaultFaces", "wall")
        self.mesh.assemble()

        probe = Probe(self.mesh)
        block = self.mesh.blocks[0]
        info = probe.get_wire_info(block.wires[wire_start][wire_end], block)

        self.assertEqual(info.starts_at_wall, starts_at_wall)
