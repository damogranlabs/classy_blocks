import copy
import time
from typing import List, Literal

import numpy as np
import scipy.optimize

from classy_blocks.mesh import Mesh
from classy_blocks.modify.clamps.clamp import ClampBase
from classy_blocks.modify.grid import Grid
from classy_blocks.modify.iteration import ClampOptimizationData, IterationDriver
from classy_blocks.modify.junction import Junction
from classy_blocks.util.constants import TOL

MinimizationMethodType = Literal["SLSQP", "L-BFGS-B", "Nelder-Mead", "Powell"]


class ParamHandler:
    N_ENTRIES = 6

    def __init__(self, junction: Junction, clamp: ClampBase, index: int):
        self.junction = junction
        self.clamp = clamp
        self.index = index

        self.sign = 1
        self.delta = TOL

        self.param_log: List[float] = []
        self.quality_log: List[float] = []

    @property
    def param(self) -> float:
        return self.clamp.params[self.index]

    @property
    def slope(self) -> float:
        return self.quality_log[-1] - self.quality_log[-2]

    @property
    def trend(self) -> float:
        return np.diff(np.diff(self.quality_log))[-1]

    @property
    def last_param(self) -> float:
        return self.param_log[-1]

    @property
    def last_delta(self) -> float:
        return self.param_log[-1] - self.param_log[-2]

    @property
    def scale(self) -> float:
        return abs(self.slope) / abs(self.last_delta + TOL)

    @property
    def done(self) -> bool:
        if len(self.quality_log) < self.N_ENTRIES:
            return False

        return self.quality_log[-3] > self.quality_log[-2] and self.quality_log[-1] > self.quality_log[-2]

    def update(self, param) -> None:
        self.clamp.params[self.index] = param
        self.clamp.update_params(self.clamp.params)

        self.param_log.append(param)

        if len(self.param_log) > self.N_ENTRIES:
            self.param_log.pop(0)

    def begin(self) -> None:
        """Sets a new parameter"""
        # make a few random tries to gather data
        if len(self.param_log) < self.N_ENTRIES:
            self.update(self.param + TOL)
            return

        if self.slope > 0:
            # quality is getting worse!
            self.sign *= -1
            self.delta /= 2
            return
        else:
            # slower improvement - larger time step
            if self.trend > 0:
                self.delta *= 0.8
            else:
                self.delta *= 1.2
        self.update(self.param_log[-1] + self.delta * self.sign)

    def end(self) -> None:
        """Calculates new quality"""
        new_quality = self.junction.quality

        if len(self.quality_log) > self.N_ENTRIES - 1 and self.quality_log[-1] < new_quality:
            # rollback!
            self.param_log.pop()
            self.quality_log.pop()
            self.delta = TOL
            self.sign *= -1
            self.update(self.param_log[-1])
            return

        self.quality_log.append(new_quality)

        if len(self.quality_log) > self.N_ENTRIES:
            self.quality_log.pop(0)


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
        # max_iterations = 2
        # driver = IterationDriver(max_iterations, tolerance)

        # start_time = time.time()

        # while not driver.converged:
        #     driver.begin_iteration(self.grid.quality)
        #     self.optimize_iteration(method)
        #     driver.end_iteration(self.grid.quality)

        # end_time = time.time()

        # if self.report:
        #     end_quality = driver.iterations[-1].final_quality
        #     start_quality = driver.iterations[0].initial_quality
        #     abs_improvement = start_quality - end_quality
        #     rel_improvement = abs_improvement / start_quality

        #     print(
        #         f"Overall improvement: {start_quality:.3e} > {end_quality:.3e}"
        #         f"({abs_improvement:.3e}, {rel_improvement*100:.0f}%)"
        #     )
        #     print(f"Elapsed time: {end_time - start_time:.0f}s")

        handlers: List[ParamHandler] = []

        for junction in self.grid.junctions:
            if junction.clamp is not None:
                for i in range(junction.clamp.dof):
                    handlers.append(ParamHandler(junction, junction.clamp, i))

        for _ in range(10000):
            for handler in handlers:
                handler.begin()

            for handler in handlers:
                handler.end()

            print(self.grid.quality)
            self.grid.clear_cache()

        import sys

        sys.exit()
