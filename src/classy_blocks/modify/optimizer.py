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

    def optimize_clamp(self, clamp: ClampBase, tolerance: float) -> None:
        junction = self._get_junction(clamp)

        report(f"  > Optimizing junction at vertex {clamp.vertex.index} ({junction.quality})")

        def fquality(params):
            # move all vertices according to X
            clamp.update_params(params)
            return self.grid.quality

        scipy.optimize.minimize(fquality, clamp.params, method="SLSQP", tol=tolerance)

        report(f"  > > Best quality: {junction.quality}")

    def optimize_iteration(self, tolerance: float) -> None:
        # gather points that can be moved with optimization
        for junction in self.grid.get_ordered_junctions():
            try:
                clamp = self._get_clamp(junction)
            except NoClampError:
                continue

            self.optimize_clamp(clamp, tolerance)

    def optimize(self, max_iterations: int = 3, tolerance: float = 1e-2) -> None:
        """Move vertices, defined and restrained in Clamps
        so that better mesh quality is obtained."""
        prev_quality = self.grid.quality

        for i in range(max_iterations):
            self.optimize_iteration(tolerance)

            this_quality = self.grid.quality

            report(f"Optimization iteration {i}: {prev_quality} > {this_quality}")

            if abs((prev_quality - this_quality) / (this_quality + VSMALL)) < tolerance:
                report("Tolerance reached, stopping optimization")
                break
