import unittest

import numpy as np
from parameterized import parameterized

from classy_blocks.base.exceptions import EdgeCreationError
from classy_blocks.construct.edges import Arc, Project
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.extrude import Extrude
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.util import functions as f
from tests.fixtures.block import BlockTestCase


class OperationTests(BlockTestCase):
    """Stuff on Operation"""

    def setUp(self):
        self.loft = self.make_loft(0)

    def test_add_side_edge_invalid_corner_index(self):
        """Fail if the user supplies an inappropriate corner to add_side_edge()"""
        with self.assertRaises(EdgeCreationError):
            self.loft.add_side_edge(4, Arc([0, 1, 0]))

    def test_set_patch_single(self):
        """Set patch of a single side"""
        self.loft.set_patch("left", "terrain")

        self.assertEqual(self.loft.patch_names["left"], "terrain")

    def test_set_patch_multiple(self):
        """Set patch of multiple sides"""
        self.loft.set_patch(["left", "bottom", "top"], "terrain")

        self.assertEqual(self.loft.patch_names["left"], "terrain")
        self.assertEqual(self.loft.patch_names["bottom"], "terrain")
        self.assertEqual(self.loft.patch_names["top"], "terrain")

    @parameterized.expand(
        (
            (0, 4, Arc),
            (0, 1, Project),
            (1, 2, Project),
            (2, 3, Project),
            (3, 0, Project),
        )
    )
    def test_edges(self, corner_1, corner_2, edge_data_class):
        """An ad-hoc Frame object with edges"""
        self.loft.project_side("bottom", "terrain", edges=True)
        self.loft.add_side_edge(0, Arc([0.1, 0.1, 0.5]))

        edges = self.loft.edges

        self.assertIsInstance(edges[corner_1][corner_2], edge_data_class)

    @parameterized.expand(("bottom", "top", "left", "right", "front", "back"))
    def test_faces(self, side):
        """A dict of fresh faces"""
        self.loft.get_face(side)

    def test_patch_from_corner_empty(self):
        """No patches defined at any corner"""
        self.assertSetEqual(self.loft.get_patches_at_corner(0), set())

    def test_patch_from_corner_single(self):
        """A single Patch at a specified corner"""
        self.loft.set_patch("bottom", "terrain")

        for i in (0, 1, 2, 3):
            self.assertSetEqual(self.loft.get_patches_at_corner(i), {"terrain"})

    def test_patch_from_corner_multiple(self):
        """Multiple patches from faces on this corner"""
        self.loft.set_patch("bottom", "terrain")
        self.loft.set_patch("front", "wall")
        self.loft.set_patch("left", "atmosphere")

        self.assertSetEqual(self.loft.get_patches_at_corner(0), {"terrain", "wall", "atmosphere"})
        self.assertSetEqual(self.loft.get_patches_at_corner(1), {"terrain", "wall"})

    def test_chop(self):
        """Chop and check"""
        self.loft.chop(0, count=10)

        self.assertEqual(len(self.loft.chops[0]), 1)

    def test_unchop(self):
        """Chop, unchop and check"""
        self.loft.chop(0, count=10)
        self.loft.unchop(0)

        self.assertEqual(len(self.loft.chops[0]), 0)

    def test_set_cell_zone(self):
        """Make sure the set_cell_zone method exists"""
        self.loft.set_cell_zone("test")

        self.assertEqual(self.loft.cell_zone, "test")

    def test_center(self):
        """Center of the operation"""
        np.testing.assert_array_almost_equal(self.loft.center, [0.5, 0.5, 0.5])

    @parameterized.expand(("bottom", "top", "front", "back", "left", "right"))
    def test_set_patch_name(self, side):
        """Set patch names and retrieve it in operation.patch_names"""
        self.loft.set_patch(side, "test")

        self.assertEqual(self.loft.patch_names[side], "test")

    @parameterized.expand(
        (
            ([0, 0, 10], "top"),
            ([0, 0, -10], "bottom"),
            ([-10, 0, 0], "left"),
            ([10, 0, 0], "right"),
            ([0, -10, 0], "front"),
            ([0, 10, 0], "back"),
        )
    )
    def test_get_closest_side(self, point, orient):
        self.assertEqual(self.loft.get_closest_side(point), orient)

    @parameterized.expand(
        (
            ([0, 0, 10],),
            ([0, 0, -10],),
            ([-10, 0, 0],),
            ([10, 0, 0],),
            ([0, -10, 0],),
            ([0, 10, 0],),
        )
    )
    def test_get_closest_face(self, point):
        face = self.loft.get_face(self.loft.get_closest_side(point))

        np.testing.assert_array_equal(face.point_array, self.loft.get_closest_face(point).point_array)

    @parameterized.expand(
        (
            ([0, 0, 10], "top"),
            ([0, 0, -10], "bottom"),
            ([-10, 0, 0], "left"),
            ([10, 0, 0], "right"),
            ([0, -10, 0], "front"),
            ([0, 10, 0], "back"),
        )
    )
    def test_get_normal_face(self, point, orient):
        normal_face = self.loft.get_normal_face(point)

        np.testing.assert_array_equal(normal_face.center, self.loft.get_face(orient).center)


class OperationProjectionTests(BlockTestCase):
    """Operation: projections"""

    def setUp(self):
        self.loft = self.make_loft(0)

    def count_edges(self, kind: str) -> int:
        """Returns the number of non-Line edges in self.loft"""
        n_edges = 0

        for beam in self.loft.edges.get_all_beams():
            edge_data = beam[2]
            if edge_data.kind == kind:
                n_edges += 1

        return n_edges

    def test_project_side_top(self):
        """Project a top face, no edges, no points"""
        self.loft.project_side("top", "terrain")

        self.assertEqual(self.loft.top_face.projected_to, "terrain")

    def test_project_side_bottom(self):
        """Project a bottom face, no edges, no points"""
        self.loft.project_side("bottom", "terrain")

        self.assertEqual(self.loft.bottom_face.projected_to, "terrain")

    @parameterized.expand(("front", "right", "back", "left"))
    def test_project_sides(self, side):
        """Project sides (without top and bottom), no points, no edges"""
        self.loft.project_side(side, "terrain")

        index = self.loft.get_index_from_side(side)
        self.assertEqual(self.loft.side_projects[index], "terrain")

    def test_no_projected_edges(self):
        """Make sure there are no other than Line edges in a plain operation"""

        self.assertEqual(self.count_edges("line"), 12)

    @parameterized.expand(("top", "bottom", "left", "right", "front", "back"))
    def test_project_sides_edges(self, side):
        """When projecting a side with edges=true, 4 Projected edges must be created"""
        self.loft.project_side(side, "terrain", edges=True)

        self.assertEqual(self.count_edges("project"), 4)

    @parameterized.expand(("top", "bottom", "left", "right", "front", "back"))
    def test_project_sides_points(self, side):
        """Project 4 points when projecting sides with points=True"""
        self.loft.project_side(side, "terrain", points=True)

        n_projected = 0

        for point in self.loft.top_face.points + self.loft.bottom_face.points:
            if point.projected_to == ["terrain"]:
                n_projected += 1

        self.assertEqual(n_projected, 4)

    def test_project_side(self):
        """Project side without edges"""
        self.loft.project_side("bottom", "terrain", edges=False)

        self.assertEqual(self.loft.bottom_face.projected_to, "terrain")

    def test_project_side_edges(self):
        """Project side with edges"""
        self.loft.project_side("bottom", "terrain", edges=True)
        self.assertEqual(self.loft.bottom_face.projected_to, "terrain")

        for edge in self.loft.bottom_face.edges:
            self.assertTrue(isinstance(edge, Project))

    def test_project_two_sides_bottom(self):
        """Project two sides with a common edge to two different geometries"""
        self.loft.project_side("bottom", "terrain", edges=True)
        self.loft.project_side("front", "walls", edges=True)

        self.assertListEqual(self.loft.bottom_face.edges[0].label, ["terrain", "walls"])

        for edge in self.loft.bottom_face.edges:
            self.assertTrue(isinstance(edge, Project))

    def test_project_two_sides_top(self):
        """Project two sides with a common edge to two different geometries"""
        self.loft.project_side("front", "terrain", edges=True)
        self.loft.project_side("top", "walls", edges=True)

        self.assertListEqual(self.loft.top_face.edges[0].label, ["terrain", "walls"])

        for edge in self.loft.top_face.edges:
            self.assertTrue(isinstance(edge, Project))

    def test_project_corner_top(self):
        """Project a vertex"""
        self.loft.project_corner(0, "terrain")

        self.assertEqual(self.loft.bottom_face.points[0].projected_to, ["terrain"])

    def test_project_corner_bottom(self):
        """Project a vertex"""
        self.loft.project_corner(4, "terrain")

        self.assertEqual(self.loft.top_face.points[0].projected_to, ["terrain"])

    @parameterized.expand(
        (
            (0, 1),  # 1
            (1, 2),  # 2
            (2, 3),  # 3
            (3, 0),  # 4
            (4, 5),  # 5
            (5, 6),  # 6
            (6, 7),  # 7
            (7, 4),  # 8
            (0, 4),  # 9
            (1, 5),  # 10
            (2, 6),  # 11
            (3, 7),  # 12
            (1, 0),  # 13
            (2, 1),  # 14
            (3, 2),  # 15
            (0, 3),  # 16
            (5, 4),  # 17
            (6, 5),  # 18
            (7, 6),  # 19
            (4, 7),  # 20
            (4, 0),  # 21
            (5, 1),  # 22
            (6, 2),  # 23
            (7, 3),  # 24
        )
    )
    def test_project_edge(self, corner_1, corner_2):
        """Find the same edge in the frame as projected pair"""
        self.loft.project_edge(corner_1, corner_2, "test")

        # find the edge in the frame and check if the corners are appropriate
        found = False

        for beam in self.loft.edges.get_all_beams():
            data = beam[2]

            if data.kind == "project":
                self.assertIn(beam[0], {corner_1, corner_2})
                self.assertIn(beam[1], {corner_1, corner_2})
                self.assertEqual(data.label, ["test"])
                found = True
                break

        self.assertTrue(found, f"Edge between {corner_1} and {corner_2} not found!")

    def test_project_edge_twice(self):
        """Project the same edge with two different geometries"""
        self.loft.project_edge(0, 1, "terrain")
        self.loft.project_edge(0, 1, "walls")

        self.assertListEqual(self.loft.bottom_face.edges[0].label, ["terrain", "walls"])

    @parameterized.expand(
        (
            ("terrain", "walls", ["terrain", "walls"]),
            ("terrain", "terrain", ["terrain"]),
            ("terrain", ["terrain", "walls"], ["terrain", "walls"]),
            (["terrain"], ["terrain"], ["terrain"]),
            (["terrain"], ["walls"], ["terrain", "walls"]),
        )
    )
    def test_project_edge_multiple(self, first, second, result):
        """Project the same edge with two equal geometries"""
        self.loft.project_edge(0, 1, first)
        self.loft.project_edge(0, 1, second)

        self.assertListEqual(self.loft.bottom_face.edges[0].label, result)


class OperationTransformTests(unittest.TestCase):
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
        top_face = bottom_face.copy().translate([0, 0, 1]).rotate(np.pi / 4, [0, 0, 1], [0, 0, 0])

        # create a mid face to take points from
        mid_face = bottom_face.copy().translate([0, 0, 0.5]).rotate(np.pi / 3, [0, 0, 1], [0, 0, 0])

        self.loft = Loft(bottom_face, top_face)

        for i, point in enumerate(mid_face.point_array):
            self.loft.add_side_edge(i, Arc(point))

    def test_construct(self):
        """Create a Loft object"""
        _ = self.loft

    def test_translate(self):
        translate_vector = np.array([0, 0, 1])

        original_op = self.loft
        translated_op = self.loft.copy().translate(translate_vector)

        np.testing.assert_almost_equal(
            original_op.bottom_face.point_array + translate_vector, translated_op.bottom_face.point_array
        )

        np.testing.assert_almost_equal(
            original_op.edges[0][4].point.position + translate_vector,
            translated_op.edges[0][4].point.position,
        )

    def test_rotate(self):
        axis = [0.0, 1.0, 0.0]
        origin = [0.0, 0.0, 0.0]
        angle = np.pi / 2

        original_op = self.loft
        rotated_op = self.loft.copy().rotate(angle, axis, origin)

        def extrude_direction(op):
            return f.unit_vector(op.top_face.point_array[0] - op.bottom_face.point_array[0])

        np.testing.assert_almost_equal(
            f.angle_between(extrude_direction(original_op), extrude_direction(rotated_op)), angle
        )

    def test_rotate_default_origin(self):
        axis = [0.0, 1.0, 0.0]
        angle = np.pi / 2

        original_op = self.loft
        rotated_op = self.loft.copy().rotate(angle, axis)

        def extrude_direction(op):
            return f.unit_vector(op.top_face.point_array[0] - op.bottom_face.point_array[0])

        np.testing.assert_almost_equal(
            f.angle_between(extrude_direction(original_op), extrude_direction(rotated_op)), angle
        )

    def test_mirror_bottom(self):
        original_loft = self.loft
        mirrored_loft = self.loft.copy().mirror([0, 0, 1])

        for i in range(4):
            # bottom faces are coincident
            np.testing.assert_almost_equal(
                original_loft.bottom_face.point_array[i], mirrored_loft.top_face.point_array[i]
            )

    def test_mirror_top(self):
        original_loft = self.loft
        mirrored_loft = self.loft.copy().mirror([0, 0, 1])

        for i in range(4):
            # top and bottom faces are exactly 2 units apart
            orig_pos = original_loft.top_face.point_array[i]
            mirrored_pos = mirrored_loft.bottom_face.point_array[i]

            np.testing.assert_almost_equal(f.norm(orig_pos - mirrored_pos), 2)

    def test_index_from_side(self):
        with self.assertRaises(RuntimeError):
            _ = self.loft.get_index_from_side("top")

    def test_mirror(self):
        extrude = Extrude(self.loft.bottom_face, [0, 0, 1])
        mirror = extrude.copy().mirror([0, 0, 1], [0, 0, 0])

        np.testing.assert_equal(extrude.bottom_face.center, mirror.top_face.center)
