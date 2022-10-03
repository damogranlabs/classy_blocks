import unittest

import numpy as np

from classy_blocks.classes.operations import Face, Extrude
from classy_blocks.util import functions as f


class OperationTests(unittest.TestCase):
    def setUp(self):
        face_points = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ]

        face_edges = [
            [0.5, -0.25, 0], None, None, None
        ]

        self.face = Face(face_points, face_edges)
        self.vector = np.array([0, 0, 1])
        self.extrude = Extrude(
            base=self.face,
            extrude_vector=self.vector
        )

    def test_translate(self):
        translate_vector = [0, 0, 1]

        original_op = self.extrude
        translated_op = self.extrude.translate(translate_vector)

        np.testing.assert_almost_equal(
            original_op.bottom_face.points + translate_vector,
            translated_op.bottom_face.points
        )

        np.testing.assert_almost_equal(
            original_op.edges[0].points + translate_vector,
            translated_op.edges[0].points
        )

    def test_rotate(self):
        axis = [0, 1, 0]
        origin = [0, 0, 0]
        angle = np.pi/2

        original_op = self.extrude
        rotated_op = self.extrude.rotate(axis, angle, origin)

        def extrude_direction(op):
            return op.top_face.points[0] - op.bottom_face.points[0]

        np.testing.assert_almost_equal(
            f.angle_between(extrude_direction(original_op),
            extrude_direction(rotated_op)),
            angle
        )

if __name__ == '__main__':
    unittest.main()