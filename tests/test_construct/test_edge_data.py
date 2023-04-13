import unittest

import numpy as np

from classy_blocks.construct import edges
from classy_blocks.base import transforms as tr
from classy_blocks.util import functions as f


class EdgeDataTests(unittest.TestCase):
    """Manipulation of edge data"""

    def test_line_create(self):
        """Create a Line edge data"""
        _ = edges.Line()

    def test_arc_transform(self):
        """Create and transform Arc edge data"""
        arc = edges.Arc([0.5, 0.25, 0])
        arc.scale(2, [0, 0, 0])

        np.testing.assert_array_almost_equal(arc.point.pos, [1, 0.5, 0])

    def test_origin_transform(self):
        """Transform Origin edge data"""
        origin = edges.Origin([1, 1, 1], 2)
        origin.transform([tr.Scaling(2, [0, 0, 0]), tr.Translation([0, -2, 0])])

        np.testing.assert_array_almost_equal(origin.origin.pos, [2, 0, 2])

    def test_angle_scale(self):
        """Axis in Angle edge must not be scaled"""
        angle = edges.Angle(1, [1, 1, 0])
        angle.scale(2, [0, 0, 0])

        # axis is normalized
        np.testing.assert_array_almost_equal(angle.axis.pos, f.unit_vector([1, 1, 0]))

    def test_angle_translate(self):
        """Axis in Angle edge must not be scaled"""
        angle = edges.Angle(1, [1, 1, 0])
        angle.translate([2, 3, 4])

        np.testing.assert_array_almost_equal(angle.axis.pos, f.unit_vector([1, 1, 0]))

    def test_angle_rotate(self):
        """Axis in angle edge must be rotated"""
        angle = edges.Angle(1, [1, 1, 0])
        angle.rotate(-np.pi / 4, [0, 0, 1], [0, 0, 0])

        np.testing.assert_array_almost_equal(angle.axis.pos, f.unit_vector([1, 0, 0]))

    def test_spline_transform(self):
        """Create and transform spline edge"""
        points = np.array([[0, 1, 0], [2**0.5 / 2, 2**0.5 / 2, 0], [1, 0, 0]])

        spline = edges.Spline(points)
        spline.scale(2, [0, 0, 0])

        np.testing.assert_array_almost_equal([p.pos for p in spline.points], 2 * points)

    def test_project_create(self):
        """Create a Projected edge"""
        _ = edges.Project(["terrain", "walls"])
