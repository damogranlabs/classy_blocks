import unittest

import numpy as np
from parameterized import parameterized

from classy_blocks.construct.operations.box import Box
from classy_blocks.mesh import Mesh
from classy_blocks.modify.clamps.free import FreeClamp
from classy_blocks.modify.clamps.links import TranslationLink
from classy_blocks.modify.find.geometric import GeometricFinder
from classy_blocks.modify.optimizer import IterationData, IterationDriver, Optimizer
from classy_blocks.util import functions as f
from classy_blocks.util.constants import VBIG


class IterationDriverTests(unittest.TestCase):
    def setUp(self):
        self.max_iterations = 20
        self.tolerance = 0.1
        self.relaxed_iterations = 2

    @property
    def driver(self) -> IterationDriver:
        return IterationDriver(self.max_iterations, self.relaxed_iterations, self.tolerance)

    @parameterized.expand(
        (
            (0, 0.5),
            (1, 0.75),
            (2, 1),
            (3, 1),
        )
    )
    def test_next_relaxation(self, iteration, relaxation):
        driver = self.driver

        for _ in range(iteration):
            driver.iterations.append(IterationData(0, 0, 0))

        self.assertEqual(driver.next_relaxation, relaxation)

    def test_no_relaxation(self):
        driver = IterationDriver(10, 0, 0.1)

        self.assertEqual(driver.next_relaxation, 1)

    def test_initial_improvement_empty(self):
        self.assertEqual(self.driver.initial_improvement, VBIG)

    def test_initial_improvement(self):
        driver = self.driver

        driver.begin_iteration(1000)
        driver.end_iteration(900)

        self.assertEqual(driver.initial_improvement, 100)

    def test_end_last_improvement_single(self):
        driver = self.driver

        driver.begin_iteration(1000)
        driver.end_iteration(900)

        self.assertEqual(driver.last_improvement, 100)

    def test_initial_improvement_multi(self):
        driver = self.driver

        driver.begin_iteration(1000)
        driver.end_iteration(900)

        driver.begin_iteration(900)
        driver.end_iteration(899)

        self.assertEqual(driver.initial_improvement, 100)

    def test_last_improvement_multi(self):
        driver = self.driver

        driver.begin_iteration(1000)
        driver.end_iteration(900)

        driver.begin_iteration(900)
        driver.end_iteration(899)

        self.assertEqual(driver.last_improvement, 1)

    def test_converged_empty(self):
        self.assertFalse(self.driver.converged)

    def test_converged_iter_limit(self):
        driver = self.driver

        for i in range(1, driver.max_iterations + 2):
            driver.begin_iteration(1000 / i)
            driver.end_iteration(1100 / i)

        self.assertTrue(driver.converged)

    def test_converged_first(self):
        """Cannot converge in the first iteration"""
        driver = self.driver

        driver.begin_iteration(1000)
        driver.end_iteration(999)

        self.assertFalse(driver.converged)

    def test_converged_relaxed(self):
        """Cannot converge until relaxation equals to 1"""
        driver = self.driver

        driver.begin_iteration(1000)
        driver.end_iteration(900)

        driver.begin_iteration(900)
        driver.end_iteration(890)

        self.assertFalse(driver.converged)

    def test_converged_inadequate(self):
        """No improvement, no convergence"""
        driver = self.driver

        for _ in range(1, driver.max_iterations - 1):
            driver.begin_iteration(1000)
            driver.end_iteration(1000)

        self.assertFalse(driver.converged)

    def test_converged_improvement(self):
        driver = self.driver

        driver.begin_iteration(1000)
        driver.end_iteration(900)

        driver.begin_iteration(900)
        driver.end_iteration(890)

        driver.begin_iteration(890)
        driver.end_iteration(889)

        self.assertTrue(driver.converged)


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

        self.vertex = next(iter(self.finder.find_in_sphere([0, 0, 0])))

    def test_optimize(self):
        # move a point, then optimize it back to
        # its initial-ish position
        self.vertex.move_to([0.3, 0.3, 0.3])

        clamp = FreeClamp(self.vertex)
        self.optimizer.release_vertex(clamp)
        self.optimizer.optimize()

        np.testing.assert_almost_equal(self.vertex.position, [0, 0, 0], decimal=1)

    def test_optimize_linked(self):
        follower = next(iter(self.finder.find_in_sphere([0, 1, 0])))
        link = TranslationLink(self.vertex, follower)

        self.vertex.move_to([0.3, 0.3, 0.3])

        clamp = FreeClamp(self.vertex)
        clamp.add_link(link)

        self.optimizer.release_vertex(clamp)
        self.optimizer.optimize()

        self.assertGreater(f.norm(follower.position - f.vector(0, 1, 0)), 0)
        np.testing.assert_almost_equal(self.vertex.position, [0, 0, 0], decimal=1)
        np.testing.assert_almost_equal(follower.position - self.vertex.position, [0, 1, 0])
