import unittest

import numpy as np
from parameterized import parameterized

from classy_blocks.optimize.links import RotationLink, SymmetryLink, TranslationLink
from classy_blocks.util import functions as f


class TranslationLinkTests(unittest.TestCase):
    def test_translate(self):
        link = TranslationLink([0, 0, 0], [1, 1, 1])

        link.leader = np.array([3, 3, 3])
        link.update()

        np.testing.assert_equal(link.follower, [4, 4, 4])


class RotationLinkTests(unittest.TestCase):
    def setUp(self):
        self.leader = np.array([1, 0, 0])
        self.follower = np.array([0, 1, 0])

    @parameterized.expand(
        (
            ([0, 0, 0],),
            ([0, 0, 1],),
            ([0, 1, 0],),
            ([1, 1, 1],),
        )
    )
    def test_radius(self, origin):
        leader = self.leader + np.array(origin)
        follower = self.follower + np.array(origin)

        link = RotationLink(leader, follower, [0, 0, 1], origin)

        np.testing.assert_equal(link._get_radius(link.leader), [1, 0, 0])

    def test_rotate(self):
        link = RotationLink(self.leader, self.follower, [0, 0, 1], [0, 0, 0])
        link.leader = np.array([0, 1, 0])
        link.update()

        np.testing.assert_equal(link.follower, [-1, 0, 0])

    def test_rotate_negative(self):
        """Rotate in negative direction"""
        link = RotationLink(self.leader, self.follower, [0, 0, 1], [0, 0, 0])

        link.leader = np.array([0, -1, 0])
        link.update()

        np.testing.assert_equal(link.follower, [1, 0, 0])

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
        orig_follower_pos = np.copy(self.follower)

        link.leader = f.rotate(link.leader, angle, axis, origin)
        link.update()

        np.testing.assert_almost_equal(link.follower, f.rotate(orig_follower_pos, angle, axis, origin))

    def test_coincident(self):
        with self.assertRaises(ValueError):
            _ = RotationLink(self.leader, self.follower, [0, 0, 1], [1, 0, 0])


class SymmetryLinkTests(unittest.TestCase):
    def test_move(self):
        link = SymmetryLink([0, 0, 1], [0, 0, -1], [0, 0, 1], [0, 0, 0])

        link.leader = np.array([0, 0, 2])
        link.update()

        np.testing.assert_equal(link.follower, [0, 0, -2])
