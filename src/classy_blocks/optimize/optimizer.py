import copy
import time
from typing import List, Literal

import numpy as np
import scipy.optimize

from classy_blocks.mesh import Mesh
from classy_blocks.optimize.clamps.clamp import ClampBase
from classy_blocks.optimize.grid import Grid
from classy_blocks.optimize.iteration import ClampOptimizationData, IterationDriver
from classy_blocks.optimize.links import LinkBase
from classy_blocks.types import IndexType, PointListType
from classy_blocks.util.constants import TOL

MinimizationMethodType = Literal["SLSQP", "L-BFGS-B", "Nelder-Mead", "Powell"]


class NoClampError(Exception):
    """Raised when there's no junction defined for a given Clamp"""


class OptimizerBase:
    """Provides tools for 2D (sketch) or 3D (mesh blocking) optimization"""

    def __init__(self, points: PointListType, addressing: List[IndexType], report: bool = True):
        self.grid = Grid(np.array(points), addressing)

        self.report = report

    def release_vertex(self, clamp: ClampBase) -> None:
        """Adds a clamp to optimization. Raises an exception if it already exists"""
        self.grid.add_clamp(clamp)

    def add_link(self, link: LinkBase) -> None:
        self.grid.add_link(link)

    def optimize_clamp(self, clamp: ClampBase, method: MinimizationMethodType) -> None:
        """Move clamp.vertex so that quality at junction is improved;
        rollback changes if grid quality decreased after optimization"""
        initial_params = copy.copy(clamp.params)
        junction = self.grid.get_junction_from_clamp(clamp)

        reporter = ClampOptimizationData(junction.index, self.grid.quality, junction.quality)
        reporter.report_start()

        def fquality(params):
            # move all vertices according to X
            clamp.update_params(params)
            self.grid.points[junction.index] = clamp.position

            if len(junction.links) > 0:
                for indexed_link in junction.links:
                    indexed_link.link.leader = clamp.position
                    indexed_link.link.update()
                    self.grid.points[indexed_link.follower_index] = indexed_link.link.follower

                return self.grid.quality

            return junction.quality

        scipy.optimize.minimize(fquality, clamp.params, bounds=clamp.bounds, method=method)

        reporter.junction_final = junction.quality
        reporter.grid_final = self.grid.quality
        reporter.report_end()

        if reporter.rollback:
            clamp.update_params(initial_params)

            if len(junction.links) > 0:
                for indexed_link in junction.links:
                    indexed_link.link.leader = clamp.position
                    indexed_link.link.update()
                    self.grid.points[indexed_link.follower_index] = indexed_link.link.follower

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
        # return np.max(np.abs(sensitivities.flatten()))

    def optimize_iteration(self, method: MinimizationMethodType) -> None:
        clamps = sorted(self.grid.clamps, key=lambda c: self._get_sensitivity(c), reverse=True)

        for clamp in clamps:
            self.optimize_clamp(clamp, method)

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


class MeshOptimizer(OptimizerBase):

    def __init__(self, mesh: Mesh, report: bool = True):
        self.mesh = mesh

        points = np.array([vertex.position for vertex in mesh.vertices])
        addressing = [block.indexes for block in self.mesh.blocks]

        super().__init__(points, addressing, report)

    def optimize(
        self, max_iterations: int = 20, tolerance: float = 0.1, method: MinimizationMethodType = "SLSQP"
    ) -> None:
        super().optimize(max_iterations, tolerance, method)

        # copy the stuff back to mesh
        for i, point in enumerate(self.grid.points):
            self.mesh.vertices[i].move_to(point)


# For backwards compatibility and ease-of-use
Optimizer = MeshOptimizer
