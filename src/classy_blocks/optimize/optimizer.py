import abc
import copy
import time
from typing import List, Literal

import numpy as np
import scipy.optimize

from classy_blocks.construct.flat.sketches.mapped import MappedSketch
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.mesh import Mesh
from classy_blocks.optimize.clamps.clamp import ClampBase
from classy_blocks.optimize.clamps.surface import PlaneClamp
from classy_blocks.optimize.grid import GridBase, HexGrid, QuadGrid
from classy_blocks.optimize.iteration import ClampOptimizationData, IterationDriver
from classy_blocks.optimize.links import LinkBase
from classy_blocks.optimize.mapper import Mapper
from classy_blocks.util.constants import TOL

MinimizationMethodType = Literal["SLSQP", "L-BFGS-B", "Nelder-Mead", "Powell"]


class OptimizerBase(abc.ABC):
    """Provides tools for 2D (sketch) or 3D (mesh blocking) optimization"""

    def __init__(self, grid: GridBase, report: bool = True):
        self.grid = grid
        self.report = report

    def add_clamp(self, clamp: ClampBase) -> None:
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
            return self.grid.update(junction.index, clamp.position)

        try:
            scipy.optimize.minimize(fquality, clamp.params, bounds=clamp.bounds, method=method)

            reporter.junction_final = junction.quality
            reporter.grid_final = self.grid.quality

            if reporter.improvement <= 0:
                reporter.rollback()

                clamp.update_params(initial_params)
                self.grid.update(junction.index, clamp.position)
        except ValueError:
            # a degenerate cell (currently) cannot be untangled;
            # try with a different junction
            reporter.skip()
            clamp.update_params(initial_params)
            self.grid.update(junction.index, clamp.position)

        reporter.report_end()

    def _get_sensitivity(self, clamp):
        """Returns maximum partial derivative at current params"""
        junction = self.grid.get_junction_from_clamp(clamp)
        initial_params = copy.copy(clamp.params)

        def fquality(clamp, junction, params):
            clamp.update_params(params)
            self.grid.update(junction.index, clamp.position)
            return junction.quality

        sensitivities = np.asarray(
            scipy.optimize.approx_fprime(clamp.params, lambda p: fquality(clamp, junction, p), epsilon=10 * TOL)
        )

        clamp.update_params(initial_params)
        self.grid.update(junction.index, clamp.position)

        return np.linalg.norm(sensitivities)

    def optimize_iteration(self, method: MinimizationMethodType) -> None:
        clamps = sorted(self.grid.clamps, key=lambda c: self._get_sensitivity(c), reverse=True)

        for clamp in clamps:
            self.optimize_clamp(clamp, method)

    def optimize(
        self, max_iterations: int = 20, tolerance: float = 0.1, method: MinimizationMethodType = "SLSQP"
    ) -> IterationDriver:
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
                f"({abs_improvement:.3e}, {rel_improvement * 100:.0f}%)"
            )
            print(f"Elapsed time: {end_time - start_time:.0f}s")

        self.backport()

        return driver

    @abc.abstractmethod
    def backport(self) -> None:
        """Reflect optimization results back to the original mesh/sketch"""


class MeshOptimizer(OptimizerBase):
    def __init__(self, mesh: Mesh, report: bool = True):
        self.mesh = mesh
        grid = HexGrid.from_mesh(self.mesh)

        super().__init__(grid, report)

    def backport(self):
        # copy the stuff back to mesh
        for i, point in enumerate(self.grid.points):
            self.mesh.vertices[i].move_to(point)


class ShapeOptimizer(OptimizerBase):
    def __init__(self, operations: List[Operation], report: bool = True):
        self.mapper = Mapper()

        for operation in operations:
            self.mapper.add(operation)

        grid = HexGrid(np.array(self.mapper.points), self.mapper.indexes)

        super().__init__(grid, report)

    def backport(self) -> None:
        # Move every point of every operation to wherever it is now
        for iop, indexes in enumerate(self.mapper.indexes):
            operation = self.mapper.elements[iop]

            for ipnt, i in enumerate(indexes):
                operation.points[ipnt].move_to(self.grid.points[i])


class SketchOptimizer(OptimizerBase):
    def __init__(self, sketch: MappedSketch, report: bool = True):
        self.sketch = sketch
        grid = QuadGrid.from_sketch(self.sketch)

        super().__init__(grid, report)

    def backport(self):
        self.sketch.update(self.grid.points)

    def auto_optimize(
        self, max_iterations: int = 20, tolerance: float = 0.1, method: MinimizationMethodType = "SLSQP"
    ) -> IterationDriver:
        """Adds a PlaneClamp to all non-boundary points and optimize the sketch.
        To include boundary points (those that can be moved along a line or a curve),
        add clamps manually before calling this method."""
        normal = self.sketch.normal

        for junction in self.grid.junctions:
            if not junction.is_boundary:
                clamp = PlaneClamp(junction.point, junction.point, normal)
                self.add_clamp(clamp)

        return super().optimize(max_iterations, tolerance, method)
