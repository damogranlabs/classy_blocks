import unittest

import numpy as np

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.edges import Arc
from classy_blocks.construct.operations.loft import Loft

from classy_blocks.util import functions as f

class OperationTests(unittest.TestCase):
    """Loft inherits directly from Operation with no additional
    whatchamacallit; Loft tests therefore validate Operation"""
    def setUp(self):
        bottom_points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        bottom_edges = [Arc([0.5, -0.25, 0]), None, None, None]

        bottom_face = Face(bottom_points, bottom_edges)
        top_face = bottom_face.copy().translate([0, 0, 1]).rotate(np.pi/4, [0, 0, 1])

        # create a mid face to take points from 
        mid_face = bottom_face.copy().translate([0, 0, 0.5]).rotate(np.pi/3, [0, 0, 1])

        self.loft = Loft(bottom_face, top_face)

        for i, point in enumerate(mid_face.points):
            self.loft.add_side_edge(i, Arc(point))

    def test_construct(self):
        """Create a Loft object"""
        _ = self.loft

    def test_translate(self):
        translate_vector = np.array([0, 0, 1])

        original_op = self.loft
        translated_op = self.loft.copy().translate(translate_vector)

        np.testing.assert_almost_equal(
            original_op.bottom_face.points + translate_vector,
            translated_op.bottom_face.points
        )

        np.testing.assert_almost_equal(
            original_op.side_edges[0].point + translate_vector,
            translated_op.side_edges[0].point
        )

    def test_rotate(self):
        axis = [0., 1., 0.]
        origin = [0., 0., 0.]
        angle = np.pi / 2

        original_op = self.loft
        rotated_op = self.loft.copy().rotate(angle, axis, origin)

        def extrude_direction(op):
            return op.top_face.points[0] - op.bottom_face.points[0]

        np.testing.assert_almost_equal(
            f.angle_between(extrude_direction(original_op), extrude_direction(rotated_op)),
            angle
        )
