import unittest
import numpy as np

from classy_blocks.construct import edges
from classy_blocks.items.vertex import Vertex
from classy_blocks.construct.flat.face import Face
from classy_blocks.util import functions as f


class FaceTests(unittest.TestCase):
    def setUp(self):
        self.points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]

    def test_face_points(self):
        # provide less than 4 points
        with self.assertRaises(Exception):
            _ = Face(self.points[:3])

    def test_face_center(self):
        np.testing.assert_array_equal(Face(self.points).center, [0.5, 0.5, 0])

    def test_translate_face(self):
        face_edges = [
            edges.Arc([0.5, -0.25, 0]),
            edges.Spline([[1.1, 0.25, 0], [1.2, 0.5, 0], [1.1, 0.75, 0]]),
            None,
            None,
        ]

        translate_vector = np.random.rand(3)

        original_face = Face(self.points, face_edges)
        translated_face = original_face.copy().translate(translate_vector)

        # check points
        for i in range(4):
            p1 = original_face.points[i]
            p2 = translated_face.points[i]

            np.testing.assert_almost_equal(p1, p2 - translate_vector)

        # check arc edge
        np.testing.assert_almost_equal(translated_face.edges[0].point - translate_vector, original_face.edges[0].point)

        # check spline edge
        for i in range(len(face_edges[1].points)):
            np.testing.assert_almost_equal(
                translated_face.edges[1].points[i] - translate_vector, original_face.edges[1].points[i]
            )

    def test_rotate_face(self):
        # only test that the Face.rotate function works properly;
        # other machinery (translate, transform...) are tested in
        # test_translate_face above
        origin = [2, 2, 2]
        angle = np.pi / 3
        axis = np.array([1, 1, 1])

        original_face = Face(self.points)
        rotated_face = original_face.copy().rotate(angle, axis, origin)

        for i in range(4):
            original_point = original_face.points[i]
            rotated_point = rotated_face.points[i]

            np.testing.assert_almost_equal(rotated_point, f.rotate(original_point, angle, axis, origin))

    def test_scale_face_default_origin(self):
        original_face = Face(self.points)
        scaled_face = original_face.scale(2)

        scaled_points = [[-0.5, -0.5, 0], [1.5, -0.5, 0], [1.5, 1.5, 0], [-0.5, 1.5, 0]]

        np.testing.assert_array_almost_equal(scaled_face.points, scaled_points)

    def test_scale_face_custom_origin(self):
        original_face = Face(self.points)
        scaled_face = original_face.scale(2, [0, 0, 0])

        scaled_points = np.array(self.points) * 2

        np.testing.assert_array_almost_equal(scaled_face.points, scaled_points)

    def test_scale_face_edges(self):
        original_edges = [edges.Arc([0.5, -0.25, 0]), None, None, None]
        original_face = Face(self.points, original_edges)

        scaled_face = original_face.scale(2, origin=[0, 0, 0])

        np.testing.assert_array_equal(scaled_face.edges[0].point, [1, -0.5, 0])
