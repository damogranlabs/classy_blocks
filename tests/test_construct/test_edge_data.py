import unittest

import numpy as np

from classy_blocks.base import transforms as tr
from classy_blocks.base.exceptions import EdgeCreationError
from classy_blocks.construct import edges
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

        np.testing.assert_array_almost_equal(arc.point.position, [1, 0.5, 0])

    def test_origin_transform(self):
        """Transform Origin edge data"""
        origin = edges.Origin([1, 1, 1], 2)
        origin.transform([tr.Scaling(2, [0, 0, 0]), tr.Translation([0, -2, 0])])

        np.testing.assert_array_almost_equal(origin.origin.position, [2, 0, 2])

    def test_angle_scale(self):
        """Axis in Angle edge must not be scaled"""
        angle = edges.Angle(1, [1, 1, 0])
        angle.scale(2, [0, 0, 0])

        # axis is normalized
        np.testing.assert_array_almost_equal(angle.axis.position, f.unit_vector([1, 1, 0]))

    def test_angle_translate(self):
        """Axis in Angle edge must not be scaled"""
        angle = edges.Angle(1, [1, 1, 0])
        angle.translate([2, 3, 4])

        np.testing.assert_array_almost_equal(angle.axis.position, f.unit_vector([1, 1, 0]))

    def test_angle_rotate(self):
        """Axis in angle edge must be rotated"""
        angle = edges.Angle(1, [1, 1, 0])
        angle.rotate(-np.pi / 4, [0, 0, 1], [0, 0, 0])

        np.testing.assert_array_almost_equal(angle.axis.position, f.unit_vector([1, 0, 0]))

    def test_spline_transform(self):
        """Create and transform spline edge"""
        points = np.array([[0, 1, 0], [2**0.5 / 2, 2**0.5 / 2, 0], [1, 0, 0]])

        spline = edges.Spline(points)
        spline.scale(2, [0, 0, 0])

        np.testing.assert_array_almost_equal(spline.curve.discretize(), 2 * points)

    def test_project_create_single(self):
        """Create an edge, projected to a single surface"""
        edge = edges.Project("terrain")

        self.assertEqual(edge.label, ["terrain"])

    def test_project_create_double(self):
        """Create an edge, projected to 2 surfaces"""
        edge = edges.Project(["terrain", "walls"])

        self.assertEqual(edge.label, ["terrain", "walls"])

    def test_project_create_multiple(self):
        """Create an edge, projected to more than 2 surfaces"""
        with self.assertRaises(EdgeCreationError):
            _ = edges.Project(["terrain", "walls", "sky"])

    def test_add_same(self):
        """Add the same label to a project edge"""
        edge = edges.Project("terrain")
        edge.add_label("terrain")

        self.assertEqual(edge.label, ["terrain"])

    def test_add_different_single(self):
        """Add a different label to a project edge"""
        edge = edges.Project("terrain")
        edge.add_label("walls")

        self.assertEqual(edge.label, ["terrain", "walls"])

    def test_add_different_list(self):
        """Add a different list of labels to a project edge"""
        edge = edges.Project("terrain")
        edge.add_label(["walls"])

        self.assertEqual(edge.label, ["terrain", "walls"])

    def test_add_too_many(self):
        """Add too many labels to a project edge"""
        edge = edges.Project("terrain")

        with self.assertRaises(EdgeCreationError):
            edge.add_label(["walls", "sky"])

    def test_default_transform(self):
        """Issue a warning error when transforming an edge
        with a default center"""
        edge = edges.Line()

        with self.assertWarns(Warning):
            edge.rotate(1, [0, 0, 1])
