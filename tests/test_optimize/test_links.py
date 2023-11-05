import unittest

import numpy as np

from classy_blocks.items.vertex import Vertex
from classy_blocks.modify.clamps.free import FreeClamp
from classy_blocks.modify.clamps.links import TranslationLink


class LinkTests(unittest.TestCase):
    def setUp(self):
        self.leader = Vertex([0, 0, 0], 0)
        self.follower = Vertex([1, 1, 1], 1)

    @property
    def link(self) -> TranslationLink:
        return TranslationLink(self.leader, self.follower)

    def test_translate(self):
        link = self.link

        self.leader.move_to([3, 3, 3])
        link.update()

        np.testing.assert_equal(self.follower.position, [4, 4, 4])

    def test_update_clamp(self):
        link = self.link
        clamp = FreeClamp(self.leader)
        clamp.add_link(link)

        clamp.update_params([3, 3, 3])

        np.testing.assert_equal(self.follower.position, [4, 4, 4])
