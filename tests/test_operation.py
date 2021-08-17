import unittest

import numpy as np

from classy_blocks.classes.operations import Face, Extrude
from classy_blocks.util import functions as f

class FaceTests(unittest.TestCase):
    def setUp(self):
        self.points = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ]   

    def test_face_points(self):
        # provide less than 4 points
        with self.assertRaises(Exception):
            Face(self.points[:3])
    
    def test_face_edges(self):
        with self.assertRaises(Exception):
            Face(
                self.points,
                [None, None, None]
            )

    def test_coplanar_points_fail(self):
        with self.assertRaises(Exception):
            self.points[-1][-1] = 0.1
            Face(self.points, check_coplanar=True)

    def test_coplanar_points_success(self):
        Face(self.points, check_coplanar=True)

    def test_translate_face(self):
        face_edges = [
            [0.5, -0.25, 0], # arc edge
            [[1.1, 0.25, 0], [1.2, 0.5, 0], [1.1, 0.75, 0]],
            None,
            None
        ]

        translate_vector = np.random.rand(3)

        original_face = Face(self.points, face_edges)
        translated_face = original_face.translate(translate_vector)

        # check points
        for i in range(4):
            p1 = original_face.points[i]
            p2 = translated_face.points[i]

            np.testing.assert_almost_equal(p1, p2 - translate_vector)

        # check arc edge
        np.testing.assert_almost_equal(
            translated_face.edges[0] - translate_vector,
            original_face.edges[0]
        )

        # check spline edge
        for i in range(len(face_edges[1])):
            np.testing.assert_almost_equal(
                translated_face.edges[1][i] - translate_vector,
                original_face.edges[1][i] 
            )

    def test_rotate_face(self):
        # only test that the Face.rotate function works properly;
        # other machinery (translate, transform_points, transform_edges) are tested in 
        # test_translate_face above
        origin = np.random.rand(3)
        angle = np.pi/3
        axis = np.array([1, 1, 1])

        original_face = Face(self.points)
        rotated_face = original_face.rotate(axis, angle, origin)
        
        for i in range(4):
            original_point = original_face.points[i]
            rotated_point = rotated_face.points[i]

            np.testing.assert_almost_equal(
                rotated_point,
                f.arbitrary_rotation(original_point, axis, angle, origin)
            )
        
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
            f.angle_between(extrude_direction(original_op), extrude_direction(rotated_op)),
            angle
        )

if __name__ == '__main__':
    unittest.main()