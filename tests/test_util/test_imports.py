import unittest

import classy_blocks as cb


class ImportsTests(unittest.TestCase):
    """Import all objects relevant to the user directly from cb"""

    def test_import_transforms(self):
        _ = cb.Translation
        _ = cb.Rotation
        _ = cb.Scaling
        _ = cb.Mirror
        _ = cb.Shear

    def test_import_curves(self):
        _ = cb.CurveBase
        _ = cb.DiscreteCurve
        _ = cb.LinearInterpolatedCurve
        _ = cb.SplineInterpolatedCurve
        _ = cb.AnalyticCurve
        _ = cb.LineCurve
        _ = cb.CircleCurve

    def test_import_edges(self):
        _ = cb.Arc
        _ = cb.Angle
        _ = cb.Origin
        _ = cb.Spline
        _ = cb.PolyLine
        _ = cb.Project
        _ = cb.OnCurve

    def test_import_flat(self):
        _ = cb.Face

    def test_import_operations(self):
        _ = cb.Loft
        _ = cb.Box
        _ = cb.Extrude
        _ = cb.Revolve
        _ = cb.Wedge
        _ = cb.Connector

    def test_import_sketches(self):
        _ = cb.MappedSketch
        _ = cb.Grid
        _ = cb.OneCoreDisk
        _ = cb.FourCoreDisk
        _ = cb.HalfDisk
        _ = cb.WrappedDisk
        _ = cb.Oval

    def test_import_stacks(self):
        _ = cb.TransformedStack
        _ = cb.ExtrudedStack
        _ = cb.RevolvedStack

    def test_import_shapes(self):
        _ = cb.Shape
        _ = cb.ExtrudedShape
        _ = cb.LoftedShape
        _ = cb.RevolvedShape
        _ = cb.Elbow
        _ = cb.Frustum
        _ = cb.Cylinder
        _ = cb.SemiCylinder
        _ = cb.ExtrudedRing
        _ = cb.RevolvedRing
        _ = cb.Hemisphere
        _ = cb.Shell

    def test_import_mesh(self):
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
        _ = cb.SymmetryLink

    def test_import_optimizer(self):
        _ = cb.MeshOptimizer
        _ = cb.ShapeOptimizer
        _ = cb.SketchOptimizer
        _ = cb.MeshSmoother
        _ = cb.SketchSmoother

    def test_import_assemblies(self):
        _ = cb.Assembly
        _ = cb.NJoint
        _ = cb.TJoint
        _ = cb.LJoint
