import copy
from typing import List

import numpy as np
import scipy.optimize

from classy_blocks.mesh import Mesh
from classy_blocks.modify.clamps.clamp import ClampBase
from classy_blocks.modify.grid import Grid
from classy_blocks.modify.junction import Junction
from classy_blocks.util.constants import VSMALL
from classy_blocks.util.tools import report


class NoJunctionError(Exception):
    """Raised when there's a clamp defined for a vertex that doesn't exist"""


class NoClampError(Exception):
    """Raised when there's no junction defined for a given Clamp"""


class Optimizer:
    """Provides tools for blocking optimization"""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.grid = Grid(mesh)

        self.clamps: List[ClampBase] = []

    def release_vertex(self, clamp: ClampBase) -> None:
        self.clamps.append(clamp)

    def _get_junction(self, clamp: ClampBase) -> Junction:
        """Returns a Junction that corresponds to clamp"""
        for junction in self.grid.junctions:
            if junction.vertex == clamp.vertex:
                return junction

        raise NoJunctionError

    def _get_clamp(self, junction: Junction) -> ClampBase:
        """Returns a Clamp that corresponds to given Junction"""
        for clamp in self.clamps:
            if clamp.vertex == junction.vertex:
                return clamp

        raise NoClampError

    def optimize_clamp(self, clamp: ClampBase, relaxation: float) -> float:
        """Move clamp.vertex so that quality at junction is improved;
        rollback changes if grid quality decreased after optimization"""
        initial_grid_quality = self.grid.quality
        initial_params = copy.copy(clamp.params)
        junction = self._get_junction(clamp)

        def fquality(params):
            # move all vertices according to X
            clamp.update_params(params)
            return junction.quality

        scipy.optimize.minimize(
            fquality,
            clamp.params,
            bounds=clamp.bounds,
            method="L-BFGS-B",
            options={"maxiter": 20, "ftol": 1, "eps": junction.delta / 10},
        )
        # alas, works well with this kind of problem but does not support bounds
        # method="COBYLA",
        # options={"maxiter": 20, "tol": 1, "rhobeg": junction.delta / 10},

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

            clamp.update_params(clamp.params, relaxation)

        return initial_grid_quality / current_grid_quality

    def optimize_iteration(self, relaxation: float) -> None:
        # gather points that can be moved with optimization
        for junction in self.grid.get_ordered_junctions():
            try:
                clamp = self._get_clamp(junction)
                self.optimize_clamp(clamp, relaxation)
            except NoClampError:
                continue

    def optimize(self, max_iterations: int = 20, tolerance: float = 0.05, initial_relaxation=0.5) -> None:
        """Move vertices, defined and restrained with Clamps
        so that better mesh quality is obtained."""
        prev_quality = self.grid.quality

        for i in range(max_iterations):
            # use lower relaxation factor with first iterations, then increase
            # TODO: tests
            relaxation = 1 - (1 - initial_relaxation) * np.exp(-i)
            self.optimize_iteration(relaxation)

            this_quality = self.grid.quality

            report(f"Optimization iteration {i}: {prev_quality:.3e} > {this_quality:.3e} (relaxation: {relaxation})")

            if abs((prev_quality - this_quality) / (this_quality + VSMALL)) < tolerance:
                report("Tolerance reached, stopping optimization")
                break

            prev_quality = this_quality
