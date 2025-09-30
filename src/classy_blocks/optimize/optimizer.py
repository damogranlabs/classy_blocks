import abc
import copy
import dataclasses
import time
from dataclasses import field
from typing import Optional

import numpy as np
import scipy.optimize

from classy_blocks.base.exceptions import OptimizationError
from classy_blocks.construct.flat.sketches.mapped import MappedSketch
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.mesh import Mesh
from classy_blocks.optimize.clamps.clamp import ClampBase
from classy_blocks.optimize.clamps.surface import PlaneClamp
from classy_blocks.optimize.grid import GridBase, HexGrid, QuadGrid
from classy_blocks.optimize.links import LinkBase
from classy_blocks.optimize.record import (
    ClampRecord,
    IterationRecord,
    MinimizationMethodType,
    OptimizationRecord,
)
from classy_blocks.optimize.report import OptimizationReporterBase, SilentReporter, TextReporter
from classy_blocks.util.constants import TOL, VSMALL


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
    # relaxation: every subsequent iteration will increase under-relaxation
    # until it's larger than relaxation_threshold; then it will fix it to 1.
    # Relaxation is identical to OpenFOAM's field relaxation
    relaxation_start: float = 1  # disabled by default
    relaxation_iterations: int = 5  # number of relaxed iterations
    relaxation_threshold: float = 0.9  # value where relaxation factor snaps to 1

    # convergence tolerance for a single joint
    # as passed to scipy.optimize.minimize
    clamp_tol: float = 1e-3
    # additional options passed to Scipy's minimize method,
    # depending on chosen algotirhm; see documentation of scipy.optimize.minimize
    # and specifically the chosen algorithm
    options: dict = field(default_factory=dict)


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

    def _get_sensitivity(self, clamp: ClampBase):
        """Returns maximum partial derivative at current params"""
        junction = self.grid.get_junction_from_clamp(clamp)
        initial_position = copy.copy(junction.point)

        def fquality(clamp, junction, params):
            try:
                clamp.update_params(params)
                quality = self.grid.update(junction.index, clamp.position)
                self.grid.update(junction.index, initial_position)

                return quality
            except Exception as e:
                print(e)
                return 0

        sensitivities = scipy.optimize.approx_fprime(clamp.params, lambda p: fquality(clamp, junction, p), epsilon=TOL)

        return np.linalg.norm(sensitivities)

    def _optimize_clamp(self, clamp: ClampBase, relaxation_factor: float) -> ClampRecord:
        """Move clamp.vertex so that quality at junction is improved;
        rollback changes if grid quality decreased after optimization"""
        junction = self.grid.get_junction_from_clamp(clamp)
        crecord = ClampRecord(junction.index, self.grid.quality)
        self.reporter.clamp_start(crecord)
        initial_position = copy.copy(junction.point)
        initial_params = copy.copy(clamp.params)

        def fquality(params):
            clamp.update_params(params)
            return self.grid.update(junction.index, clamp.position)

        try:
            result = scipy.optimize.minimize(
                fquality,
                clamp.params,
                bounds=clamp.bounds,
                method=self.config.method,
                tol=self.config.clamp_tol,
                options=self.config.options,
            )
            if not result.success:
                raise OptimizationError(result.message)

            # relax and update
            for i, param in enumerate(result.x):
                clamp.params[i] = initial_params[i] + relaxation_factor * (param - initial_params[i])
            fquality(clamp.params)

            # always check grid quality, not clamp's
            crecord.grid_final = self.grid.quality

            if not crecord.improvement > 0:
                raise OptimizationError("No improvement")
        except OptimizationError as e:
            # roll back to the initial state
            self.grid.update(junction.index, initial_position)
            crecord.rolled_back = True
            crecord.error_message = str(e)
            crecord.grid_final = self.grid.quality

        self.reporter.clamp_end(crecord)

        return crecord

    def _optimize_iteration(self, iteration_no: int) -> IterationRecord:
        rlf = self.relaxation_factor(iteration_no)
        irecord = IterationRecord(iteration_no, self.grid.quality, rlf)
        self.reporter.iteration_start(iteration_no, rlf)

        clamps = sorted(self.grid.clamps, key=lambda c: self._get_sensitivity(c), reverse=True)
        for clamp in clamps:
            self._optimize_clamp(clamp, rlf)

        irecord.grid_final = self.grid.quality
        self.reporter.iteration_end(irecord)

        return irecord

    def relaxation_factor(self, iteration_no: int) -> float:
        iter_no = iteration_no
        threshold = self.config.relaxation_threshold
        start_relax = self.config.relaxation_start
        target_iter = self.config.relaxation_iterations

        if iter_no >= target_iter:
            return 1.0
        if start_relax >= threshold:
            return 1.0

        k = -np.log(1 - (threshold - start_relax) / (threshold - start_relax + VSMALL))

        # Normalize iteration to [0, 1]
        t = iter_no / target_iter

        # increase the factor slowly at the beginning and quicker at the end
        # Slow start, fast finish
        value = start_relax + (threshold - start_relax) * (1 - np.exp(-k * (t**3)))

        return value

    def optimize(
        self,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
        method: Optional[MinimizationMethodType] = None,
    ) -> bool:
        """Move vertices as defined and restrained with Clamps
        so that better mesh quality is obtained.

        Within each iteration, all vertices will be moved, starting with the one with the most influence on quality.
        Lower tolerance values.

        max_iterations, tolerance (relative) and method enable rough adjustment of optimization;
        for fine tuning, modify optimizer.config attribute.

        Returns True is optimization was successful (tolerance reached)"""
        if max_iterations is not None:
            self.config.max_iterations = max_iterations
        if tolerance is not None:
            self.config.rel_tol = tolerance
        if method is not None:
            self.config.method = method

        orecord = OptimizationRecord(time.time(), self.grid.quality)  # TODO: cache repeating quality queries

        for i in range(self.config.max_iterations):
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
    def __init__(self, operations: list[Operation], report: bool = True, merge_tol: float = TOL):
        grid = HexGrid.from_elements(operations, merge_tol)

        super().__init__(grid, report)
        self.operations = operations

    def _backport(self) -> None:
        # Move every point of every operation to wherever it is now
        for iop, indexes in enumerate(self.grid.addressing):
            operation = self.operations[iop]

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
        self,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
        method: Optional[MinimizationMethodType] = None,
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
