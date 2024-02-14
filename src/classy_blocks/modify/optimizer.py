import copy
import dataclasses
from typing import ClassVar, List

import numpy as np
import scipy.optimize

from classy_blocks.mesh import Mesh
from classy_blocks.modify.clamps.clamp import ClampBase
from classy_blocks.modify.grid import Grid
from classy_blocks.modify.junction import Junction
from classy_blocks.util.constants import TOL, VBIG, VSMALL
from classy_blocks.util.tools import report


class NoJunctionError(Exception):
    """Raised when there's a clamp defined for a vertex that doesn't exist"""


class NoClampError(Exception):
    """Raised when there's no junction defined for a given Clamp"""


class ClampExistsError(Exception):
    """Raised when attempting to add a vertex that's already among existing clamps"""


@dataclasses.dataclass
class IterationData:
    """Data about a single iteration's progress"""

    index: int
    relaxation: float
    initial_quality: float
    final_quality: float = VBIG

    @property
    def improvement(self) -> float:
        if abs(self.initial_quality - self.final_quality) < VSMALL:
            return VSMALL

        return self.initial_quality - self.final_quality

    def report_begin(self):
        report(f"Starting iteration {self.index+1} (relaxation {self.relaxation:.2f})")

    def report_end(self):
        report(f"Iteration {self.index+1} finished. Improvement: {self.initial_quality:.3e} > {self.final_quality:.3e}")


class IterationDriver:
    """Bookkeeping: iterations, results, relaxation, quality and whatnot"""

    INITIAL_RELAXATION: ClassVar[float] = 0.5

    def __init__(self, max_iterations: int, relaxed_iterations: int, tolerance: float):
        self.max_iterations = max_iterations
        self.relaxed_iterations = relaxed_iterations
        self.tolerance = tolerance

        self.iterations: List[IterationData] = []

    def begin_iteration(self, quality: float) -> IterationData:
        iteration = IterationData(len(self.iterations), self.next_relaxation, quality)
        iteration.report_begin()

        self.iterations.append(iteration)

        return iteration

    def end_iteration(self, quality: float) -> None:
        iteration = self.iterations[-1]

        iteration.final_quality = quality
        iteration.report_end()

    @property
    def next_relaxation(self) -> float:
        """Returns the relaxation factor for the next iteration"""
        if self.relaxed_iterations == 0:
            return 1

        step = (1 - self.INITIAL_RELAXATION) / self.relaxed_iterations
        iteration = len(self.iterations)

        return min(1, IterationDriver.INITIAL_RELAXATION + step * iteration)

    @property
    def initial_improvement(self) -> float:
        if len(self.iterations) < 1:
            return VBIG

        return self.iterations[0].improvement

    @property
    def last_improvement(self) -> float:
        if len(self.iterations) < 2:
            return self.initial_improvement

        return self.iterations[-1].improvement

    @property
    def converged(self) -> bool:
        if len(self.iterations) <= 1:
            # At least two iterations are needed
            # so that the result of the last can be compared with the first one
            return False

        if len(self.iterations) <= self.relaxed_iterations:
            # always make at least relaxed_iterations + 1
            # so that the last iteration has relaxation = 1
            return False

        if len(self.iterations) >= self.max_iterations:
            report("Iteration limit hit, stopping optimization.")
            return True

        if self.last_improvement / self.initial_improvement < self.tolerance * self.initial_improvement:
            print("Tolerance reached, stopping optimization.")
            return True

        return False


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

    def optimize_clamp(self, clamp: ClampBase, iteration: IterationData) -> float:
        """Move clamp.vertex so that quality at junction is improved;
        rollback changes if grid quality decreased after optimization"""
        initial_grid_quality = self.grid.quality
        initial_params = copy.copy(clamp.params)
        junction = self._get_junction(clamp)

        def fquality(params):
            # move all vertices according to X
            clamp.update_params(params)

            if clamp.is_linked:
                return self.grid.quality
            return junction.quality

        scipy.optimize.minimize(fquality, clamp.params, bounds=clamp.bounds, method="SLSQP")

        current_grid_quality = self.grid.quality

        if current_grid_quality > initial_grid_quality:
            # rollback if quality is worse
            clamp.update_params(initial_params)
            msg = (
                f"  < Rollback at vertex {clamp.vertex.index}: {initial_grid_quality:.3e} < {current_grid_quality:.3e}"
            )
            report(msg)
            current_grid_quality = 1
        else:
            msg = "  > Optimized junction at vertex "
            msg += f"{clamp.vertex.index}: {initial_grid_quality:.3e} > {current_grid_quality:.3e}"
            report(msg)

            clamp.update_params(clamp.params, iteration.relaxation)

        return initial_grid_quality / current_grid_quality

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

    def optimize_iteration(self, iteration: IterationData) -> None:
        self.clamps.sort(key=lambda c: self._get_sensitivity(c), reverse=True)

        for clamp in self.clamps:
            self.optimize_clamp(clamp, iteration)
        self.clamps.sort(key=lambda c: self._get_sensitivity(c), reverse=True)

        for clamp in self.clamps:
            self.optimize_clamp(clamp, iteration)

    def optimize(self, max_iterations: int = 20, relaxed_iterations: int = 2, tolerance: float = 0.1) -> None:
        """Move vertices, defined and restrained with Clamps
        so that better mesh quality is obtained.

        Within each iteration, all vertices will be moved, starting with the one with the most influence on quality.
        Lower tolerance values"""
        driver = IterationDriver(max_iterations, relaxed_iterations, tolerance)

        while not driver.converged:
            iteration = driver.begin_iteration(self.grid.quality)
            self.optimize_iteration(iteration)
            driver.end_iteration(self.grid.quality)
