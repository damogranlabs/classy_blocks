import unittest

import numpy as np

from classy_blocks.construct.operations.box import Box
from classy_blocks.mesh import Mesh
from classy_blocks.modify.clamps.free import FreeClamp
from classy_blocks.modify.clamps.links import TranslationLink
from classy_blocks.modify.find.geometric import GeometricFinder
from classy_blocks.modify.optimizer import Optimizer
from classy_blocks.util import functions as f


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.mesh = Mesh()

        # generate a cube, consisting of 2x2x2 smaller cubes
        for x in (-1, 0):
            for y in (-1, 0):
                for z in (-1, 0):
                    box = Box([x, y, z], [x + 1, y + 1, z + 1])

                    for axis in range(3):
                        box.chop(axis, count=10)

                    self.mesh.add(box)

        self.mesh.assemble()

        self.finder = GeometricFinder(self.mesh)
        self.optimizer = Optimizer(self.mesh)

        self.vertex = list(self.finder.find_in_sphere([0, 0, 0]))[0]

    def test_optimize(self):
        # move a point, then optimize it back to
        # its initial-ish position
        self.vertex.move_to([0.3, 0.3, 0.3])

        clamp = FreeClamp(self.vertex)
        self.optimizer.release_vertex(clamp)
        self.optimizer.optimize()

        np.testing.assert_almost_equal(self.vertex.position, [0, 0, 0], decimal=1)

    def test_optimize_linked(self):
        follower = list(self.finder.find_in_sphere([0, 1, 0]))[0]
        link = TranslationLink(self.vertex, follower)

        self.vertex.move_to([0.3, 0.3, 0.3])

        clamp = FreeClamp(self.vertex)
        clamp.add_link(link)

        self.optimizer.release_vertex(clamp)
        self.optimizer.optimize()

        self.assertGreater(f.norm(follower.position - f.vector(0, 1, 0)), 0)
        np.testing.assert_almost_equal(self.vertex.position, [0, 0, 0], decimal=1)
        np.testing.assert_equal(follower.position - self.vertex.position, [0, 1, 0])
