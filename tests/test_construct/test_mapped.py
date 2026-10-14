import dataclasses
import json
import os
import tempfile
import unittest

import numpy as np
import trimesh

from classy_blocks.base.exceptions import MappedShapeCreationError
from classy_blocks.construct.shapes.mapped.description import MappedShapeDescription
from classy_blocks.construct.shapes.mapped.mapped import MappedShape
from classy_blocks.construct.surfaces.triangulated import TriangulatedSurface
from classy_blocks.mesh import Mesh
from tests.fixtures import data
from tests.test_construct.test_mapped_features import HEXAS, POINTS, blocking, box_sides, surfaces


class MappedShapeTests(unittest.TestCase):
    def setUp(self):
        self.description = MappedShapeDescription(
            data.fl + data.cl,
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [1, 8, 9, 2, 5, 10, 11, 6],
                [2, 9, 12, 13, 6, 11, 14, 15],
            ],
            {},
            {},
        )

    def test_simple(self):
        # a simple 3-block shape
        mesh = Mesh()
        mesh.add(MappedShape(self.description))

        mesh.assemble()

        self.assertEqual(len(mesh.dump.blocks), 3)


class MappedShapeConformTests(unittest.TestCase):
    def shape(self, names: dict[str, str]) -> MappedShape:
        """Two blocks, a bit off the 2x1x1 box, fitted to it"""
        description = blocking(HEXAS, names)
        moved = description.points * 1.1 + np.random.default_rng(0).uniform(-0.05, 0.05, (len(POINTS), 3))

        return MappedShape(dataclasses.replace(description, points=moved, surfaces=surfaces(box_sides())))

    def test_points_land_on_the_box(self):
        points = np.concatenate([loft.point_array for loft in self.shape({}).operations])

        expected = np.concatenate([POINTS[hexa] for hexa in HEXAS])
        # the middle points are free to slide along their edges
        sliding = np.isclose(expected[:, 0], 1)

        np.testing.assert_allclose(points[~sliding], expected[~sliding], atol=1e-6)
        np.testing.assert_allclose(points[sliding, 1:], expected[sliding, 1:], atol=1e-6)

    def test_optimization_keeps_points_on_the_box(self):
        shape = self.shape({})
        shape.optimize(report=False)

        _, distances, _ = trimesh.proximity.closest_point(
            trimesh.creation.box(bounds=[[0, 0, 0], [2, 1, 1]]), shape.points
        )
        np.testing.assert_allclose(distances, 0, atol=1e-6)
        np.testing.assert_allclose(shape.points[[0, 2, 6, 8]], POINTS[[0, 2, 6, 8]], atol=1e-6)

    def test_patches(self):
        loft = self.shape({}).operations[0]

        self.assertEqual(loft.patch_names["front"], "miny")

    def test_sides_are_projected(self):
        loft = self.shape({}).operations[0]

        self.assertEqual(loft.side_projects[0], "miny")

    def test_curve_edges_are_projected_to_both_surfaces(self):
        loft = self.shape({}).operations[0]

        self.assertEqual(loft.bottom_face.edges[0].label, ["miny", "minz"])

    def test_shared_edge(self):
        """The edge between blocks lies on a single surface"""
        first, second = self.shape({}).operations

        self.assertEqual(first.side_edges[1].label, ["miny"])

    def test_geometry(self):
        self.assertEqual(set(self.shape({}).geometry), set(box_sides()))

    def test_a_mismatch_is_reported(self):
        with self.assertRaises(MappedShapeCreationError):
            self.shape({"minx": "maxy"})


class MappedShapeDescriptionInnerTests(unittest.TestCase):
    def test_no_inner_points(self):
        self.assertEqual(blocking(HEXAS, {}).inner, [])

    def test_a_point_in_the_middle(self):
        """Eight blocks around the middle of a 3x3x3 lattice of points"""
        grid = np.indices((3, 3, 3)).reshape(3, -1).T
        index = {tuple(point): i for i, point in enumerate(grid)}
        corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        hexas = [
            [index[(x + dx, y + dy, z + dz)] for dx, dy, dz in corners]
            for x in range(2)
            for y in range(2)
            for z in range(2)
        ]

        self.assertEqual(MappedShapeDescription(grid, hexas, {}, {}).inner, [index[(1, 1, 1)]])


class MappedShapeDescriptionLoadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "cube.shape")
        self.stl = os.path.join(self.directory, "top.stl")
        trimesh.creation.box().export(self.stl)

        content = {
            "points": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
            "hexas": [list(range(8))],
            "patches": {"top": [[4, 5, 6, 7]]},
            "surfaces": {"top": "top.stl"},
        }
        with open(self.path, "w") as handle:
            json.dump(content, handle)

    def test_topology(self):
        description = MappedShapeDescription.load(self.path)

        np.testing.assert_allclose(description.points[6], [1, 1, 1])
        self.assertEqual(description.hexas, [list(range(8))])
        self.assertEqual(description.patches, {"top": [[4, 5, 6, 7]]})

    def test_surface_relative_to_file(self):
        surface = MappedShapeDescription.load(self.path).surfaces["top"]

        # trimesh's default box spans -0.5 .. 0.5
        np.testing.assert_allclose(surface.get_closest_point([10, 0, 0])[0], 0.5)

    def test_surface_remembers_file(self):
        surface = MappedShapeDescription.load(self.path).surfaces["top"]

        self.assertIsInstance(surface, TriangulatedSurface)
        self.assertEqual(surface.path, self.stl)

    def test_missing_surface(self):
        os.remove(self.stl)

        with self.assertRaises(ValueError):
            MappedShapeDescription.load(self.path)
