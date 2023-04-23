import unittest

import numpy as np

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.extrude import Extrude
from classy_blocks.construct.edges import Arc
from classy_blocks.util import functions as f


class ExtrudeTests(unittest.TestCase):
    def setUp(self):
        self.points = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        self.edges = [Arc([0.5, 0.1, 0]), None, None, None]

        self.amount = [0, 0, 1]

    @property
    def face(self) -> Face:
        return Face(self.points, self.edges)

    @property
    def extrude(self) -> Extrude:
        return Extrude(self.face, self.amount)

    def test_extrude_box(self):
        """Create an Extrude"""
        ext = self.extrude

        for i in range(4):
            np.testing.assert_array_almost_equal(
                ext.top_face.points[i].position - ext.bottom_face.points[i].position, self.amount
            )

    def test_extrude_slanted(self):
        """Extrude with a different vector"""
        self.amount = [1, 1, 1]

        ext = self.extrude

        for i in range(4):
            np.testing.assert_array_almost_equal(
                ext.top_face.points[i].position - ext.bottom_face.points[i].position, self.amount
            )

    def test_extrude_edges(self):
        """Test that extrude copies edges"""
        n_arc = 0

        for data in self.extrude.edges.get_all_beams():
            if data is not None:
                edge = data[2]

                if edge.kind == "arc":
                    n_arc += 1

        self.assertEqual(n_arc, 2)

    def test_extrude_amount(self):
        """Extrude by amount"""
        self.points = np.asarray(self.points) / 2
        self.amount = 0.1

        ext = self.extrude

        for i in range(4):
            self.assertAlmostEqual(
                f.norm(ext.top_face.points[i].position - ext.bottom_face.points[i].position), self.amount
            )
