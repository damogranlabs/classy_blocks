import abc
import copy
import dataclasses
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
from classy_blocks.optimize.links import LinkBase
from classy_blocks.optimize.mapper import Mapper
from classy_blocks.util.constants import TOL
from classy_blocks.util.tools import report

MinimizationMethodType = Literal["SLSQP", "L-BFGS-B", "Nelder-Mead", "Powell"]


@dataclasses.dataclass
class OptimizationData:
    # maximum number of iterations; the optimizer will quit at the last iteration
    # regardless of achieved quality
    max_iterations: int = 20
    # absolute tolerance; if overall grid quality stays within given interval
    # between iterations the optimizer will quit
    abs_tol: float = 10
    # relative tolerance; if relative change between iterations is less than
    # given, the optimizer will quit
    rel_tol: float = 0.01
    # method that will be used for scipy.optimize.minimize
    # (only those that support all required features are valid)
    method: MinimizationMethodType = "SLSQP"
    # relaxation base; every subsequent iteration will increase under-relaxation
    # until it's larger than relaxation_threshold; then it will fix it to 1.
    # Relaxation is identical to OpenFOAM's field relaxation
    relaxation: float = 1
    relaxation_threshold: float = 0.9

    # print more-or-less pretty info about optimization
    report: bool = True
    # convergence tolerance for a single joint
    # as passed to scipy.optimize.minimize
    clamp_tol: float = 1e-3


@dataclasses.dataclass
class OptimizationRecord:
    vertex_index: int
    grid_initial: float = -1
    junction_initial: float = -1
    junction_final: float = -1
    grid_final: float = -1

    rolled_back: bool = False


class OptimizationReporter:
    COL_W = 15

    def __init__(self) -> None:
        # TODO: add typing and move to a separate file
        self.time_start = None

    def start_optimization(self):
        self.time_start = time.time()

    def start_iteration(self, iteration_no: int) -> None:
        report(f"Optimization iteration {iteration_no}")
        headers = ["Vertex", "Initial", "Local", "Improvement", "Final", "Status"]
        for h in headers:
            report(f"{h:>{self.COL_W}s}")

    def report_start(self) -> None:
        report(f"{self.index:>6}", end="   ")
        report(f"{self.grid_initial:.3e}", end="   ")
        report(f"{self.junction_initial:.3e}", end="   ")

    def report_end(self) -> None:
        report(f"{self.improvement: >11.0f}", end="   ")
        report(f"{self.grid_final:.3e}", end="   ")

    def start_clamp(self):
        pass

    def end_clamp(self):
        pass

    def end_iteration(self):
        pass

    def end_optimization(self):
        pass


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

    def optimize_clamp(self, clamp: ClampBase, data: OptimizationData, relaxation_factor: float) -> OptimizationRecord:
        """Move clamp.vertex so that quality at junction is improved;
        rollback changes if grid quality decreased after optimization"""
        junction = self.grid.get_junction_from_clamp(clamp)
        record = OptimizationRecord(junction.index)
        record.grid_initial = self.grid.quality
        record.junction_initial = junction.quality

        initial_params = copy.copy(clamp.params)

        def fquality(params):
            clamp.update_params(params)
            quality = self.grid.update(junction.index, clamp.position)
            return quality

        try:
            results = scipy.optimize.minimize(
                fquality, clamp.params, bounds=clamp.bounds, method=data.method, tol=data.clamp_tol
            )
            if not results.success:
                # TODO! Differentiate different fail modes and report them in output
                raise ValueError

            for i, param in enumerate(results.x):
                clamp.params[i] = initial_params[i] + relaxation_factor * (param - initial_params[i])

            # update the position after relaxation
            record.junction_final = fquality(clamp.params)
            record.grid_final = self.grid.quality

            improvement = record.grid_initial - record.grid_final
            if np.isnan(improvement) or improvement <= 0:
                raise ValueError
        except ValueError:
            clamp.update_params(initial_params)
            record.rolled_back = True

        self.grid.update(junction.index, clamp.position)
        return record

    def optimize_iteration(self, data: OptimizationData, relaxation_factor: float) -> None:
        for clamp in self.grid.clamps:
            self.optimize_clamp(clamp, data, relaxation_factor)

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

        TODO:
        Returns True is optimization was successful (tolerance reached)"""
        data = OptimizationData(max_iterations=max_iterations, rel_tol=tolerance, method=method, **kwargs)
        last_quality = self.grid.quality

        for i in range(max_iterations):
            rlf = self.relaxation_factor(data, i)
            print(f"Optimizing {i}, relaxation: {rlf}")
            self.optimize_iteration(data, rlf)
            this_quality = self.grid.quality

            if last_quality - this_quality < data.abs_tol:
                break
            last_quality = this_quality
        else:
            self.backport()
            return False

        self.backport()
        return True  # TODO: return properly

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
