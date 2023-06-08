import numpy as np

from classy_blocks.base import transforms as tr
from classy_blocks.items.vertex import Vertex
from classy_blocks.util import constants
from tests.fixtures.data import DataTestCase


class VertexTests(DataTestCase):
    """Vertex object"""

    def test_assert_3d(self):
        """Raise an exception if the point is not in 3D space"""
        with self.assertRaises(AssertionError):
            _ = Vertex([0, 0], 0)

    def test_translate_int(self):
        """Vertex translation with integer delta"""
        delta = [1.0, 0.0, 0.0]

        np.testing.assert_array_almost_equal(Vertex([0, 0, 0], 0).translate(delta).position, delta)

    def test_translate_float(self):
        """Vertex translation with float delta"""
        coords = [0.0, 0.0, 0.0]
        delta = [1.0, 0.0, 0.0]

        np.testing.assert_array_almost_equal(Vertex(coords, 0).translate(delta).position, delta)

    def test_rotate(self):
        """Rotate a point"""
        coords = [1.0, 0.0, 0.0]

        np.testing.assert_array_almost_equal(
            Vertex(coords, 0).rotate(np.pi / 2, [0, 0, 1], [0, 0, 0]).position, [0, 1, 0]
        )

    def test_scale(self):
        """Scale a point"""
        coords = [1.0, 0.0, 0.0]

        np.testing.assert_array_almost_equal(Vertex(coords, 0).scale(2, [0, 0, 0]).position, [2, 0, 0])

    def test_inequal(self):
        """The __eq__ method returns False"""
        point_1 = Vertex([0, 0, 0], 0)
        point_2 = Vertex([0, 0, 0 + 2 * constants.TOL], 1)

        self.assertFalse(point_1 == point_2)

    def test_description_plain(self):
        """A Rudimentary Vertex description"""
        v = Vertex([0, 0, 0], 0)

        self.assertEqual(v.description, "(0.00000000 0.00000000 0.00000000) // 0")

    def test_project_single(self):
        """Add a single geometry to project to"""
        v = Vertex([0.0, 0.0, 0.0], 0)
        v.project("terrain")

        expected = "project (0.00000000 0.00000000 0.00000000) (terrain) // 0"

        self.assertEqual(v.description, expected)

    def test_project_multiple(self):
        """Add a single geometry to project to"""
        v = Vertex([0.0, 0.0, 0.0], 0)
        v.project(["terrain", "walls", "border"])

        expected = "project (0.00000000 0.00000000 0.00000000) (terrain walls border) // 0"

        self.assertEqual(v.description, expected)

    def test_multitransform(self):
        """Use the Transformation class for multiple transforms"""
        v = Vertex([0, 1, 1], 0)

        v.transform(
            [tr.Rotation([0, 0, 1], -np.pi / 2, [0, 0, 0]), tr.Scaling(2, [0, 0, 0]), tr.Translation([-2, 0, -2])]
        )

        np.testing.assert_array_almost_equal(v.position, [0, 0, 0])
