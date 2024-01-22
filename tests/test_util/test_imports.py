import unittest

import classy_blocks as cb


class ImportsTests(unittest.TestCase):
    """Import all objects relevant to the user directly from cb"""

    def test_import_curves(self):
        _ = cb.DiscreteCurve
        _ = cb.LinearInterpolatedCurve
        _ = cb.SplineInterpolatedCurve
        _ = cb.AnalyticCurve

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
        _ = cb.OnCurve

    def test_import_operations(self):
        """Import Operations"""
        _ = cb.Loft
        _ = cb.Box
        _ = cb.Extrude
        _ = cb.Revolve
        _ = cb.Wedge
        _ = cb.Connector

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

    def test_import_finders(self):
        _ = cb.GeometricFinder
        _ = cb.RoundSolidFinder

    def test_import_clamps(self):
        _ = cb.ClampBase
        _ = cb.FreeClamp
        _ = cb.LineClamp
        _ = cb.CurveClamp
        _ = cb.RadialClamp

    def test_import_links(self):
        _ = cb.LinkBase
        _ = cb.TranslationLink
        _ = cb.RotationLink

    def test_import_optimizer(self):
        _ = cb.Optimizer
