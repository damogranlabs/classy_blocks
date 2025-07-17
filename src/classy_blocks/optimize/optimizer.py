import abc
import copy
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
    OptimizationData,
    OptimizationRecord,
)
from classy_blocks.optimize.report import OptimizationReporterBase, SilentReporter, TextReporter
from classy_blocks.util.constants import TOL


class OptimizerBase(abc.ABC):
    """Provides tools for 2D (sketch) or 3D (mesh blocking) optimization"""

    reporter: OptimizationReporterBase

    def __init__(self, grid: GridBase, report: bool = True):
        self.grid = grid

        if not report:
            self.reporter = SilentReporter()
        else:
            self.reporter = TextReporter()

    def add_clamp(self, clamp: ClampBase) -> None:
        """Adds a clamp to optimization. Raises an exception if it already exists"""
        self.grid.add_clamp(clamp)

    def add_link(self, link: LinkBase) -> None:
        self.grid.add_link(link)

    def optimize_clamp(self, clamp: ClampBase, data: OptimizationData, relaxation_factor: float) -> ClampRecord:
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
                fquality, clamp.params, bounds=clamp.bounds, method=data.method, tol=data.clamp_tol
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

    def optimize_iteration(self, data: OptimizationData, iteration_no: int) -> IterationRecord:
        rlf = self.relaxation_factor(data, iteration_no)
        irecord = IterationRecord(iteration_no, self.grid.quality, rlf)
        self.reporter.iteration_start(iteration_no)

        for clamp in self.grid.clamps:
            self.optimize_clamp(clamp, data, rlf)

        irecord.grid_final = self.grid.quality
        self.reporter.iteration_end(irecord)

        return irecord

    @staticmethod
    def relaxation_factor(data: OptimizationData, iteration_no: int) -> float:
        if data.relaxation == 1:
            # relaxation disabled
            return 1
        relax_base = 1 / data.relaxation
        relaxation_factor = 1 - relax_base ** -(iteration_no + 1)
        # it makes no sense to relax after positions have been fixed approximately
        if relaxation_factor > data.relaxation_threshold:
            relaxation_factor = 1

        return relaxation_factor

    def optimize(
        self, max_iterations: int = 20, tolerance: float = 0.1, method: MinimizationMethodType = "SLSQP", **kwargs
    ) -> bool:
        """Move vertices as defined and restrained with Clamps
        so that better mesh quality is obtained.

        Within each iteration, all vertices will be moved, starting with the one with the most influence on quality.
        Lower tolerance values.

        Returns True is optimization was successful (tolerance reached)"""
        data = OptimizationData(max_iterations=max_iterations, rel_tol=tolerance, method=method, **kwargs)
        orecord = OptimizationRecord(time.time(), self.grid.quality)  # TODO: cache repeating quality queries

        for i in range(max_iterations):
            iter_record = self.optimize_iteration(data, i)

            if iter_record.abs_improvement < data.abs_tol:
                orecord.termination = "abs"
                break
            if iter_record.rel_improvement < data.rel_tol:
                orecord.termination = "rel"
                break
        else:
            orecord.termination = "limit"

        orecord.grid_final = self.grid.quality
        orecord.time_end = time.time()
        self.reporter.optimization_end(orecord)
        self.backport()

        return orecord.termination in ("abs", "rel")

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
    def __init__(self, operations: List[Operation], report: bool = True, merge_tol: float = TOL):
        self.mapper = Mapper(merge_tol)

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

        grid = QuadGrid(sketch.positions, sketch.indexes)

        super().__init__(grid, report)

    def backport(self):
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
