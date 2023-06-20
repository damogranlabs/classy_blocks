import unittest

import classy_blocks as cb


class ImportsTests(unittest.TestCase):
    """Import all objects relevant to the user directly from cb"""

    def test_import_flat(self):
        """Flat stuff"""
        _ = cb.Face

    def test_import_edges(self):
        """Import edge data"""
        _ = cb.Arc
        _ = cb.Angle
        _ = cb.Origin
        _ = cb.Spline
        _ = cb.PolyLine
        _ = cb.Project

    def test_import_operations(self):
        """Import Operations"""
        _ = cb.Loft
        _ = cb.Box
        _ = cb.Extrude
        _ = cb.Revolve
        _ = cb.Wedge

    def test_import_shapes(self):
        """Import Shapes"""
        _ = cb.Elbow
        _ = cb.Frustum
        _ = cb.Cylinder
        _ = cb.ExtrudedRing
        _ = cb.RevolvedRing
        _ = cb.Hemisphere
        _ = cb.Shell

    def test_import_mesh(self):
        """The core stuff"""
        _ = cb.Mesh
