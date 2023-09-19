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

    def optimize_clamp(self, clamp: ClampBase) -> float:
        initial_quality = self.grid.quality
        initial_params = copy.copy(clamp.params)

        def fquality(params):
            # move all vertices according to X
            clamp.update_params(params)
            return self.grid.quality

        scipy.optimize.minimize(fquality, clamp.params, method="SLSQP", tol=1e-2, options={"maxiter": 100})

        current_quality = self.grid.quality

        if current_quality > initial_quality:
            # rollback if quality is worse
            clamp.update_params(initial_params)
            current_quality = 1

        report(f"  > Optimized junction at vertex {clamp.vertex.index}: {initial_quality} > {self.grid.quality}")

        return initial_quality / current_quality

    def optimize_iteration(self) -> None:
        # gather points that can be moved with optimization
        for junction in self.grid.get_ordered_junctions():
            try:
                clamp = self._get_clamp(junction)
                self.optimize_clamp(clamp)
            except NoClampError:
                continue

    def optimize(self, max_iterations: int = 20, tolerance: float = 0.05) -> None:
        """Move vertices, defined and restrained with Clamps
        so that better mesh quality is obtained."""
        prev_quality = self.grid.quality

        for i in range(max_iterations):
            self.optimize_iteration()

            this_quality = self.grid.quality

            report(f"Optimization iteration {i}: {prev_quality} > {this_quality}")

            if abs((prev_quality - this_quality) / (this_quality + VSMALL)) < tolerance:
                report("Tolerance reached, stopping optimization")
                break

            prev_quality = this_quality
