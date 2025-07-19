import abc
import copy
import dataclasses
import time
from typing import List

import numpy as np
import scipy.optimize

from classy_blocks.construct.flat.sketches.mapped import MappedSketch
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.mesh import Mesh
from classy_blocks.optimize.clamps.clamp import ClampBase
from classy_blocks.optimize.clamps.surface import PlaneClamp
from classy_blocks.optimize.grid import GridBase, HexGrid, QuadGrid
from classy_blocks.optimize.links import LinkBase
from classy_blocks.optimize.mapper import Mapper
from classy_blocks.optimize.record import (
    ClampRecord,
    IterationRecord,
    MinimizationMethodType,
    OptimizationRecord,
)
from classy_blocks.optimize.report import OptimizationReporterBase, SilentReporter, TextReporter
from classy_blocks.util.constants import TOL


@dataclasses.dataclass
class OptimizerConfig:
    # maximum number of iterations; the optimizer will quit at the last iteration
    # regardless of achieved quality
    max_iterations: int = 20
    # absolute tolerance; if overall grid quality stays within given interval
    # between iterations the optimizer will quit
    abs_tol: float = 0  # disabled by default
    # relative tolerance; if relative change between iterations is less than
    # given, the optimizer will quit
    rel_tol: float = 0.01
    # method that will be used for scipy.optimize.minimize
    # (only those that support all required features are valid)
    method: MinimizationMethodType = "SLSQP"
    # relaxation base; every subsequent iteration will increase under-relaxation
    # until it's larger than relaxation_threshold; then it will fix it to 1.
    # Relaxation is identical to OpenFOAM's field relaxation
    relaxation: float = 1  # disabled by default
    relaxation_threshold: float = 0.9

    # convergence tolerance for a single joint
    # as passed to scipy.optimize.minimize
    clamp_tol: float = 1e-3


class OptimizerBase(abc.ABC):
    """Provides tools for 2D (sketch) or 3D (mesh blocking) optimization"""

    reporter: OptimizationReporterBase

    def __init__(self, grid: GridBase, report: bool = True):
        self.grid = grid

        if not report:
            self.reporter = SilentReporter()
        else:
            self.reporter = TextReporter()

        # holds defaults and can be adjusted before calling .optimize()
        self.config = OptimizerConfig()

    def add_clamp(self, clamp: ClampBase) -> None:
        """Adds a clamp to optimization. Raises an exception if it already exists"""
        self.grid.add_clamp(clamp)

    def add_link(self, link: LinkBase) -> None:
        self.grid.add_link(link)

    def _optimize_clamp(self, clamp: ClampBase, relaxation_factor: float) -> ClampRecord:
        """Move clamp.vertex so that quality at junction is improved;
        rollback changes if grid quality decreased after optimization"""
        junction = self.grid.get_junction_from_clamp(clamp)
        crecord = ClampRecord(junction.index, self.grid.quality, junction.quality)
        self.reporter.clamp_start(crecord)
        initial_params = copy.copy(clamp.params)

        def fquality(params):
            clamp.update_params(params)
            quality = self.grid.update(junction.index, clamp.position)
            return quality

        try:
            result = scipy.optimize.minimize(
                fquality, clamp.params, bounds=clamp.bounds, method=self.config.method, tol=self.config.clamp_tol
            )
            if not result.success:
                raise ValueError(result.message)

            # relax and update
            for i, param in enumerate(result.x):
                clamp.params[i] = initial_params[i] + relaxation_factor * (param - initial_params[i])

            crecord.grid_final = fquality(clamp.params)

            if np.isnan(crecord.improvement) or crecord.improvement <= 0:
                raise ValueError("No improvement")
        except ValueError as e:
            # roll back to the initial state
            fquality(initial_params)
            crecord.rolled_back = True
            crecord.error_message = str(e)

        crecord.junction_final = junction.quality
        crecord.grid_final = self.grid.quality
        self.reporter.clamp_end(crecord)

        return crecord

    def _optimize_iteration(self, iteration_no: int) -> IterationRecord:
        rlf = self.relaxation_factor(iteration_no)
        irecord = IterationRecord(iteration_no, self.grid.quality, rlf)
        self.reporter.iteration_start(iteration_no)

        for clamp in self.grid.clamps:
            self._optimize_clamp(clamp, rlf)

        irecord.grid_final = self.grid.quality
        self.reporter.iteration_end(irecord)

        return irecord

    def relaxation_factor(self, iteration_no: int) -> float:
        if self.config.relaxation == 1:
            # relaxation disabled
            return 1
        relax_base = 1 / self.config.relaxation
        relaxation_factor = 1 - relax_base ** -(iteration_no + 1)
        # it makes no sense to relax after positions have been fixed approximately
        if relaxation_factor > self.config.relaxation_threshold:
            relaxation_factor = 1

        return relaxation_factor

    def optimize(
        self,
        max_iterations: int = 20,
        tolerance: float = 0.1,
        method: MinimizationMethodType = "SLSQP",
    ) -> bool:
        """Move vertices as defined and restrained with Clamps
        so that better mesh quality is obtained.

        Within each iteration, all vertices will be moved, starting with the one with the most influence on quality.
        Lower tolerance values.

        max_iterations, tolerance (relative) and method enable rough adjustment of optimization;
        for fine tuning, modify optimizer.config attribute.

        Returns True is optimization was successful (tolerance reached)"""
        self.config.max_iterations = max_iterations
        self.config.rel_tol = tolerance
        self.config.method = method
        orecord = OptimizationRecord(time.time(), self.grid.quality)  # TODO: cache repeating quality queries

        for i in range(max_iterations):
            iter_record = self._optimize_iteration(i)

            if iter_record.abs_improvement < self.config.abs_tol:
                orecord.termination = "abs"
                break
            if iter_record.rel_improvement < self.config.rel_tol:
                orecord.termination = "rel"
                break
        else:
            orecord.termination = "limit"

        orecord.grid_final = self.grid.quality
        orecord.time_end = time.time()
        self.reporter.optimization_end(orecord)
        self._backport()

        return orecord.termination in ("abs", "rel")

    @abc.abstractmethod
    def _backport(self) -> None:
        """Reflect optimization results back to the original mesh/sketch"""


class MeshOptimizer(OptimizerBase):
    def __init__(self, mesh: Mesh, report: bool = True):
        self.mesh = mesh
        grid = HexGrid.from_mesh(self.mesh)

        super().__init__(grid, report)

    def _backport(self):
        # copy the stuff back to mesh
        for i, point in enumerate(self.grid.points):
            self.mesh.vertices[i].move_to(point)


class ShapeOptimizer(OptimizerBase):
    def __init__(self, operations: List[Operation], report: bool = True, merge_tol: float = TOL):
        self.mapper = Mapper(merge_tol)

        for operation in operations:
            self.mapper.add(operation)

        grid = HexGrid(np.array(self.mapper.points), self.mapper.indexes)

        super().__init__(grid, report)

    def _backport(self) -> None:
        # Move every point of every operation to wherever it is now
        for iop, indexes in enumerate(self.mapper.indexes):
            operation = self.mapper.elements[iop]

            for ipnt, i in enumerate(indexes):
                operation.points[ipnt].move_to(self.grid.points[i])


class SketchOptimizer(OptimizerBase):
    def __init__(self, sketch: MappedSketch, report: bool = True):
        self.sketch = sketch

        grid = QuadGrid(sketch.positions, sketch.indexes)

        super().__init__(grid, report)

    def _backport(self):
        self.sketch.update(self.grid.points)

    def auto_optimize(
        self, max_iterations: int = 20, tolerance: float = 0.1, method: MinimizationMethodType = "SLSQP"
    ) -> bool:
        """Adds a PlaneClamp to all non-boundary points and optimize the sketch.
        To include boundary points (those that can be moved along a line or a curve),
        add clamps manually before calling this method."""
        normal = self.sketch.normal

        for junction in self.grid.junctions:
            if not junction.is_boundary:
                clamp = PlaneClamp(junction.point, junction.point, normal)
                self.add_clamp(clamp)

        return super().optimize(max_iterations, tolerance, method)
