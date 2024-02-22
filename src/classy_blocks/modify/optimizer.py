import copy
from typing import List

import numpy as np
import scipy.optimize

from classy_blocks.mesh import Mesh
from classy_blocks.modify.clamps.clamp import ClampBase
from classy_blocks.modify.grid import Grid
from classy_blocks.modify.iteration import ClampOptimizationData, IterationDriver
from classy_blocks.modify.junction import Junction
from classy_blocks.util.constants import TOL


class NoJunctionError(Exception):
    """Raised when there's a clamp defined for a vertex that doesn't exist"""


class NoClampError(Exception):
    """Raised when there's no junction defined for a given Clamp"""


class ClampExistsError(Exception):
    """Raised when attempting to add a vertex that's already among existing clamps"""


class Optimizer:
    """Provides tools for blocking optimization"""

    def __init__(self, mesh: Mesh, report: bool = True):
        self.mesh = mesh
        self.report = report

        self.grid = Grid(mesh)
        self.clamps: List[ClampBase] = []

    def release_vertex(self, clamp: ClampBase) -> None:
        """Adds a clamp to optimization. Raises an exception if it already exists"""
        for existing in self.clamps:
            if existing.vertex == clamp.vertex:
                raise ClampExistsError(f"A clamp has already been defined for vertex {existing}")
        self.clamps.append(clamp)

    def _get_junction(self, clamp: ClampBase) -> Junction:
        """Returns a Junction that corresponds to clamp"""
        for junction in self.grid.junctions:
            if junction.vertex == clamp.vertex:
                return junction

        raise NoJunctionError

    def optimize_clamp(self, clamp: ClampBase) -> None:
        """Move clamp.vertex so that quality at junction is improved;
        rollback changes if grid quality decreased after optimization"""
        initial_params = copy.copy(clamp.params)
        junction = self._get_junction(clamp)

        reporter = ClampOptimizationData(clamp.vertex.index, self.grid.quality, junction.quality)
        reporter.report_start()

        def fquality(params):
            # move all vertices according to X
            clamp.update_params(params)

            if clamp.is_linked:
                return self.grid.quality
            return junction.quality

        scipy.optimize.minimize(fquality, clamp.params, bounds=clamp.bounds, method="SLSQP")

        reporter.junction_final = junction.quality
        reporter.grid_final = self.grid.quality
        reporter.report_end()

        if reporter.rollback:
            clamp.update_params(initial_params)

    def _get_sensitivity(self, clamp):
        """Returns maximum partial derivative at current params"""
        junction = self._get_junction(clamp)

        def fquality(clamp, junction, params):
            clamp.update_params(params)
            return junction.quality

        sensitivities = np.asarray(
            scipy.optimize.approx_fprime(clamp.params, lambda p: fquality(clamp, junction, p), epsilon=10 * TOL)
        )
        return np.max(np.abs(sensitivities.flatten()))

    def optimize_iteration(self) -> None:
        self.clamps.sort(key=lambda c: self._get_sensitivity(c), reverse=True)

        for clamp in self.clamps:
            self.optimize_clamp(clamp)

    def optimize(self, max_iterations: int = 20, tolerance: float = 0.1) -> None:
        """Move vertices, defined and restrained with Clamps
        so that better mesh quality is obtained.

        Within each iteration, all vertices will be moved, starting with the one with the most influence on quality.
        Lower tolerance values"""
        driver = IterationDriver(max_iterations, tolerance)

        while not driver.converged:
            driver.begin_iteration(self.grid.quality)
            self.optimize_iteration()
            driver.end_iteration(self.grid.quality)
