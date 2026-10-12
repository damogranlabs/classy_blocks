import json
import os
import tempfile
import unittest

import numpy as np
import trimesh

from classy_blocks.construct.shapes.mapped.description import MappedShapeDescription
from classy_blocks.construct.shapes.mapped.mapped import MappedShape
from classy_blocks.construct.surfaces.triangulated import TriangulatedSurface
from classy_blocks.mesh import Mesh
from tests.fixtures import data


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
