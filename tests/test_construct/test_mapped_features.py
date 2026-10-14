import os
import tempfile
import unittest
from collections import defaultdict
from typing import Optional

import numpy as np
import trimesh

from classy_blocks.construct.shapes.mapped.association import Kind, Label, associate
from classy_blocks.construct.shapes.mapped.description import MappedShapeDescription
from classy_blocks.construct.shapes.mapped.features import Features
from classy_blocks.construct.surfaces.triangulated import TriangulatedSurface
from classy_blocks.optimize.cell import HexCell

# a box's sides, named by the axis and end they face, and their outward normals
SIDES = {f"{end}{axis}": (index, sign) for index, axis in enumerate("xyz") for end, sign in (("min", -1), ("max", 1))}


def split(mesh: trimesh.Trimesh, masks: dict[str, np.ndarray]) -> dict[str, trimesh.Trimesh]:
    return {name: mesh.submesh([np.where(mask)[0]], append=True) for name, mask in masks.items()}


def box_sides(names: Optional[dict[str, str]] = None) -> dict[str, trimesh.Trimesh]:
    """The 2x1x1 box's sides, each named after itself or what names maps it to"""
    box = trimesh.creation.box(bounds=[[0, 0, 0], [2, 1, 1]])
    masks: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(box.faces), dtype=bool))
    for side, (axis, sign) in SIDES.items():
        masks[(names or {}).get(side, side)] |= np.isclose(box.face_normals[:, axis], sign)

    return split(box, masks)


def surfaces(meshes: dict[str, trimesh.Trimesh]) -> dict[str, TriangulatedSurface]:
    return {name: TriangulatedSurface(mesh) for name, mesh in meshes.items()}


def pipe(caps: dict[str, float]) -> dict[str, TriangulatedSurface]:
    """A cylinder's side and its caps, grouped by their normals' z component"""
    cylinder = trimesh.creation.cylinder(radius=1, height=2, sections=16)
    normals = cylinder.face_normals[:, 2]
    masks = {"side": np.isclose(normals, 0), **{name: np.isin(np.round(normals), z) for name, z in caps.items()}}

    return surfaces(split(cylinder, masks))


# two blocks along x in the 2x1x1 box; the first one alone spans x = 0..1
BOTTOM = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]]
POINTS = np.array(BOTTOM + [[x, y, 1] for x, y, _ in BOTTOM], dtype=float)
HEXAS = [[0, 1, 4, 3, 6, 7, 10, 9], [1, 2, 5, 4, 7, 8, 11, 10]]


# the box's corners alone: one block spanning it
ONE_BLOCK = [[0, 2, 5, 3, 6, 8, 11, 9]]


def blocking(hexas: list[list[int]], names: dict[str, str]) -> MappedShapeDescription:
    """Block sides on the 2x1x1 box's boundary go to the patch named after their box side,
    or to the one names maps that side to"""
    faces = [[hexa[i] for i in indexes] for hexa in hexas for indexes in HexCell.side_indexes]
    patches: dict[str, list[list[int]]] = defaultdict(list)
    for side, (axis, sign) in SIDES.items():
        value = (POINTS[:, axis].min(), POINTS[:, axis].max())[sign > 0]
        patches[names.get(side, side)] += [face for face in faces if np.allclose(POINTS[face][:, axis], value)]

    return MappedShapeDescription(POINTS, hexas, dict(patches), {})


def associated(description: MappedShapeDescription, features: Features) -> list[Label]:
    return associate(description.points, description.patches, features)


def kinds(description: MappedShapeDescription, features: Features) -> list[Kind]:
    return [label.kind for label in associated(description, features)]


class FeaturesTests(unittest.TestCase):
    def test_box(self):
        features = Features.from_surfaces(surfaces(box_sides()))

        self.assertEqual(len(features.curves), 12)
        self.assertEqual(len(features.corners), 8)
        np.testing.assert_allclose(features.corners[frozenset(["maxx", "maxy", "maxz"])][0].point, [2, 1, 1])

    def test_a_curve_runs_along_the_edge(self):
        (curve,) = Features.from_surfaces(surfaces(box_sides())).curves[frozenset(["minx", "miny"])]

        np.testing.assert_allclose(sorted(curve.series.points[:, 2]), [0, 1])
        np.testing.assert_allclose(curve.series.points[:, :2], 0)

    def test_a_curve_point_projects_between_its_vertices(self):
        (curve,) = Features.from_surfaces(surfaces(box_sides())).curves[frozenset(["minx", "miny"])]

        np.testing.assert_allclose(curve.get_closest_point([-1, -1, 0.3]), [0, 0, 0.3], atol=1e-6)

    def test_pipe_ends_are_closed_curves(self):
        features = Features.from_surfaces(pipe({"top": 1, "bottom": -1}))
        curve = features.curves[frozenset(["side", "top"])][0].series.points

        self.assertEqual(len(features.curves), 2)
        self.assertEqual(features.corners, {})
        np.testing.assert_allclose(curve[0], curve[-1])
        np.testing.assert_allclose(np.linalg.norm(curve[:, :2], axis=1), 1)

    def test_surfaces_meeting_twice_make_two_curves(self):
        curves = Features.from_surfaces(pipe({"ends": [1, -1]})).curves[frozenset(["ends", "side"])]

        self.assertEqual(sorted(curve.discretize()[0, 2] for curve in curves), [-1, 1])

    def test_surfaces_meeting_twice_make_two_curves_and_corners(self):
        features = Features.from_surfaces(surfaces(box_sides({"minx": "ends", "maxx": "ends"})))

        self.assertEqual(len(features.curves[frozenset(["ends", "miny"])]), 2)
        points = [corner.point for corner in features.corners[frozenset(["ends", "miny", "minz"])]]
        np.testing.assert_allclose(sorted(points, key=lambda point: point[0]), [[0, 0, 0], [2, 0, 0]])

    def test_no_surfaces(self):
        features = Features.from_surfaces({})

        self.assertEqual((features.curves, features.corners), ({}, {}))

    def test_find_by_names(self):
        features = Features.from_surfaces(surfaces(box_sides()))

        self.assertIs(features.find(frozenset(["minx"]), [0, 0, 0]), features.surfaces["minx"])
        self.assertIs(
            features.find(frozenset(["minx", "miny"]), [0, 0, 0]), features.curves[frozenset(["minx", "miny"])][0]
        )
        self.assertIsNone(features.find(frozenset(["minx", "maxx"]), [0, 0, 0]))

    def test_the_nearest_piece_is_found(self):
        features = Features.from_surfaces(pipe({"ends": [1, -1]}))
        curve = features.find(frozenset(["ends", "side"]), [0, 0, 0.9])

        np.testing.assert_allclose(curve.discretize()[:, 2], 1)


class MultiSolidTests(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "box.stl")
        text = "".join(
            trimesh.exchange.stl.export_stl_ascii(mesh).replace("solid", f"solid {name}", 1) + "\n"
            for name, mesh in box_sides().items()
        )
        with open(self.path, "w") as handle:
            handle.write(text)

    def test_the_whole_file_without_a_name(self):
        self.assertEqual(len(TriangulatedSurface.from_file(self.path).mesh.faces), 12)


class MultiplePiecesTests(unittest.TestCase):
    def test_points_stick_to_their_own_corner(self):
        features = Features.from_surfaces(surfaces(box_sides({"minx": "ends", "maxx": "ends"})))
        labels = associated(blocking(HEXAS, {"minx": "ends", "maxx": "ends"}), features)

        np.testing.assert_allclose([labels[0].target.point, labels[2].target.point], [[0, 0, 0], [2, 0, 0]])


class AssociationTests(unittest.TestCase):
    def setUp(self):
        self.features = Features.from_surfaces(surfaces(box_sides()))

    def test_one_block_sits_on_the_corners(self):
        labels = kinds(blocking(ONE_BLOCK, {}), self.features)

        self.assertEqual(labels.count(Kind.CORNER), 8)
        self.assertEqual(labels.count(Kind.FREE), 4)  # the points no block uses

    def test_two_blocks_add_curve_points(self):
        labels = kinds(blocking(HEXAS, {}), self.features)

        self.assertEqual((labels.count(Kind.CORNER), labels.count(Kind.CURVE)), (8, 4))

    def test_a_patch_without_a_surface_leaves_its_points_free(self):
        labels = associated(blocking(HEXAS, {"minx": "inlet"}), self.features)

        self.assertEqual(labels[0].names, frozenset(["miny", "minz"]))
        self.assertEqual(labels[0].kind, Kind.CURVE)

    def test_a_wrong_surface_is_a_mismatch(self):
        labels = associated(blocking(HEXAS, {"minx": "maxy"}), self.features)

        self.assertEqual(labels[0].kind, Kind.FREE)
        self.assertEqual(labels[0].error, "No feature where maxy, miny, minz meet")
        self.assertEqual(labels[1].error, "")
