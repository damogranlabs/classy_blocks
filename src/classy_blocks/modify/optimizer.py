import copy
import time
from typing import Literal

import numpy as np
import scipy.optimize

from classy_blocks.mesh import Mesh
from classy_blocks.modify.clamps.clamp import ClampBase
from classy_blocks.modify.grid import Grid
from classy_blocks.modify.iteration import ClampOptimizationData, IterationDriver
from classy_blocks.modify.junction import Junction
from classy_blocks.util.constants import TOL

MinimizationMethodType = Literal["SLSQP", "L-BFGS-B", "Nelder-Mead", "Powell"]


class Optimizer:
    """Provides tools for blocking optimization"""

    def __init__(self, mesh: Mesh, report: bool = True):
        self.mesh = mesh
        self.report = report

        self.grid = Grid(mesh)

    def release_vertex(self, clamp: ClampBase) -> None:
        """Adds a clamp to optimization. Raises an exception if it already exists"""
        self.grid.add_clamp(clamp)

    def optimize_junction(self, junction: Junction, method: MinimizationMethodType) -> None:
        """Move clamp.vertex so that quality at junction is improved;
        rollback changes if grid quality decreased after optimization"""
        if junction.clamp is None:
            raise ValueError(f"No clamp at this junction {junction.vertex.index}")

        clamp = junction.clamp
        initial_params = copy.copy(clamp.params)

        reporter = ClampOptimizationData(clamp.vertex.index, self.grid.quality, junction.quality)
        reporter.report_start()

        def fquality(params):
            # move all vertices according to X
            clamp.update_params(params)

            if clamp.is_linked:
                return self.grid.quality
            return junction.quality

        scipy.optimize.minimize(fquality, clamp.params, bounds=clamp.bounds, method=method)

        reporter.junction_final = junction.quality
        reporter.grid_final = self.grid.quality
        reporter.report_end()

        if reporter.rollback:
            clamp.update_params(initial_params)

    def _get_sensitivity(self, clamp):
        """Returns maximum partial derivative at current params"""
        junction = self.grid.get_junction_from_clamp(clamp)

        def fquality(clamp, junction, params):
            clamp.update_params(params)
            return junction.quality

        sensitivities = np.asarray(
            scipy.optimize.approx_fprime(clamp.params, lambda p: fquality(clamp, junction, p), epsilon=10 * TOL)
        )
        return np.linalg.norm(sensitivities)

    def optimize_iteration(self, method: MinimizationMethodType) -> None:
        junctions = []

        for junction in self.grid.junctions:
            if junction.clamp is None:
                continue
            junctions.append(junction)

        junctions.sort(key=lambda j: self._get_sensitivity(j.clamp), reverse=True)

        for junction in junctions:
            self.optimize_junction(junction, method)

    def optimize(
        self, max_iterations: int = 20, tolerance: float = 0.1, method: MinimizationMethodType = "SLSQP"
    ) -> None:
        """Move vertices, defined and restrained with Clamps
        so that better mesh quality is obtained.

        Within each iteration, all vertices will be moved, starting with the one with the most influence on quality.
        Lower tolerance values"""
        driver = IterationDriver(max_iterations, tolerance)

        start_time = time.time()

        while not driver.converged:
            driver.begin_iteration(self.grid.quality)
            self.optimize_iteration(method)
            driver.end_iteration(self.grid.quality)

        end_time = time.time()

        if self.report:
            end_quality = driver.iterations[-1].final_quality
            start_quality = driver.iterations[0].initial_quality
            abs_improvement = start_quality - end_quality
            rel_improvement = abs_improvement / start_quality

            print(
                f"Overall improvement: {start_quality:.3e} > {end_quality:.3e}"
                f"({abs_improvement:.3e}, {rel_improvement*100:.0f}%)"
            )
            print(f"Elapsed time: {end_time - start_time:.0f}s")
