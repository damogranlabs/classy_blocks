import copy
from typing import List

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
            method="COBYLA",
            options={"maxiter": 10, "tol": 1, "rhobeg": 1e-4},
        )

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

            # underrelaxation
            import numpy as np

            old_params = np.array(initial_params)
            new_params = np.array(clamp.params)
            relaxed = old_params + relaxation * (new_params - old_params)

            clamp.update_params(relaxed)

        return initial_grid_quality / current_grid_quality

    def optimize_iteration(self, iteration: int) -> None:
        # gather points that can be moved with optimization
        if iteration > 0:
            relaxation = 1
        else:
            relaxation = 0.5

        for junction in self.grid.get_ordered_junctions():
            try:
                clamp = self._get_clamp(junction)
                self.optimize_clamp(clamp, relaxation)
            except NoClampError:
                continue

    def optimize(self, max_iterations: int = 20, tolerance: float = 0.05) -> None:
        """Move vertices, defined and restrained with Clamps
        so that better mesh quality is obtained."""
        prev_quality = self.grid.quality

        for i in range(max_iterations):
            self.optimize_iteration(i)

            this_quality = self.grid.quality

            report(f"Optimization iteration {i}: {prev_quality:.3e} > {this_quality:.3e}")

            if abs((prev_quality - this_quality) / (this_quality + VSMALL)) < tolerance:
                report("Tolerance reached, stopping optimization")
                break

            prev_quality = this_quality
