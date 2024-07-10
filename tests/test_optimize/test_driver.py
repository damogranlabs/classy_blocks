import unittest

from classy_blocks.optimize.iteration import IterationDriver
from classy_blocks.util.constants import VBIG


class IterationDriverTests(unittest.TestCase):
    def setUp(self):
        self.max_iterations = 20
        self.tolerance = 0.1

    @property
    def driver(self) -> IterationDriver:
        return IterationDriver(self.max_iterations, self.tolerance)

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

    def test_converged_inadequate(self):
        """No improvement, no convergence"""
        driver = self.driver
        driver.tolerance = 0

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
