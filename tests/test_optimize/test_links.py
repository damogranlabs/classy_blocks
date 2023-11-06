import unittest

import numpy as np

from classy_blocks.items.vertex import Vertex
from classy_blocks.modify.clamps.free import FreeClamp
from classy_blocks.modify.clamps.links import RotationLink, TranslationLink


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

    def test_rotate(self):
        link = RotationLink(self.leader, self.follower, [0, 0, 0], [0, 0, 1])

        self.leader.move_to([0, 1, 0])
        link.update()

        np.testing.assert_equal(self.follower.position, [-1, 0, 0])
