import unittest

import numpy as np
from parameterized import parameterized

from classy_blocks.items.vertex import Vertex
from classy_blocks.modify.clamps.free import FreeClamp
from classy_blocks.modify.clamps.links import RotationLink, TranslationLink
from classy_blocks.util import functions as f


class TranslationLinkTests(unittest.TestCase):
    def setUp(self):
        self.leader = Vertex([0, 0, 0], 0)
        self.follower = Vertex([1, 1, 1], 1)

    def test_translate(self):
        link = TranslationLink(self.leader, self.follower)

        self.leader.move_to([3, 3, 3])
        link.update()

        np.testing.assert_equal(self.follower.position, [4, 4, 4])

    def test_update_clamp(self):
        link = TranslationLink(self.leader, self.follower)
        clamp = FreeClamp(self.leader)
        clamp.add_link(link)

        clamp.update_params([3, 3, 3])

        np.testing.assert_equal(self.follower.position, [4, 4, 4])


class RotationLinkTests(unittest.TestCase):
    def setUp(self):
        self.leader = Vertex([1, 0, 0], 0)
        self.follower = Vertex([0, 1, 0], 1)

    @parameterized.expand(
        (
            ([0, 0, 0],),
            ([0, 0, 1],),
            ([0, 1, 0],),
            ([1, 1, 1],),
        )
    )
    def test_radius(self, origin):
        self.leader.translate(origin)
        self.follower.translate(origin)

        link = RotationLink(self.leader, self.follower, [0, 0, 1], origin)

        np.testing.assert_equal(link._get_radius(self.leader.position), [1, 0, 0])

    def test_rotate(self):
        link = RotationLink(self.leader, self.follower, [0, 0, 1], [0, 0, 0])

        self.leader.move_to([0, 1, 0])
        link.update()

        np.testing.assert_equal(self.follower.position, [-1, 0, 0])

    def test_rotate_negative(self):
        """Rotate in negative direction"""
        link = RotationLink(self.leader, self.follower, [0, 0, 1], [0, 0, 0])

        self.leader.move_to([0, -1, 0])
        link.update()

        np.testing.assert_equal(self.follower.position, [1, 0, 0])

    @parameterized.expand(
        (
            # z-axis, origin
            ([0, 0, 1], [0, 0, 0], np.pi / 3),
            ([0, 0, 1], [0, 0, 0], np.pi / 4),
            ([0, 0, 1], [0, 0, 0], np.pi / 6),
            ([0, 0, 1], [0, 0, 0], -np.pi / 3),
            ([0, 0, 1], [0, 0, 0], -np.pi / 4),
            ([0, 0, 1], [0, 0, 0], -np.pi / 6),
            # -z axis, origin
            ([0, 0, -1], [0, 0, 0], np.pi / 3),
            ([0, 0, -1], [0, 0, 0], np.pi / 4),
            ([0, 0, -1], [0, 0, 0], np.pi / 6),
            ([0, 0, -1], [0, 0, 0], -np.pi / 3),
            ([0, 0, -1], [0, 0, 0], -np.pi / 4),
            ([0, 0, -1], [0, 0, 0], -np.pi / 6),
            # skewed axis, origin
            ([0, 1, 1], [0, 0, 0], np.pi / 2),
            ([0, 1, 1], [0, 0, 0], np.pi / 3),
            ([0, 1, 1], [0, 0, 0], np.pi / 4),
            ([0, 1, 1], [0, 0, 0], np.pi / 6),
            # skewed axis, different origin
            ([0, 1, 1], [-1, -1, -1], np.pi / 2),
            ([0, 1, 1], [-1, -1, -1], np.pi / 3),
            ([0, 1, 1], [-1, -1, -1], np.pi / 4),
            ([0, 1, 1], [-1, -1, -1], np.pi / 6),
            # negative angles
            ([0, 1, 1], [-1, -1, -1], -np.pi / 2),
            ([0, 1, 1], [-1, -1, -1], -np.pi / 3),
            ([0, 1, 1], [-1, -1, -1], -np.pi / 4),
            ([0, 1, 1], [-1, -1, -1], -np.pi / 6),
        )
    )
    def test_rotate_arbitrary(self, axis, origin, angle):
        link = RotationLink(self.leader, self.follower, axis, origin)
        orig_follower_pos = np.copy(self.follower.position)

        self.leader.rotate(angle, axis, origin)
        link.update()

        np.testing.assert_almost_equal(self.follower.position, f.rotate(orig_follower_pos, angle, axis, origin))

    def test_update(self):
        # Update an unchanged link and check that it's the same
        self.follower.translate([1, 0, 1])
        link = RotationLink(self.leader, self.follower, [0, 0, 1], [0, 0, 0])

        link.update()

        np.testing.assert_almost_equal(self.follower.position, [1, 1, 1])

    def test_coincident(self):
        with self.assertRaises(ValueError):
            _ = RotationLink(self.leader, self.follower, [0, 0, 1], [1, 0, 0])
