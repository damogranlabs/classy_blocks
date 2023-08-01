from typing import List

from classy_blocks.mesh import Mesh
from classy_blocks.modify.clamps.clamp import ClampBase
from classy_blocks.modify.grid import Grid


class Optimizer:
    """Provides tools for blocking optimization"""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.grid = Grid(mesh)

        self.clamps: List[ClampBase] = []

    def release_vertex(self, clamp: ClampBase) -> None:
        self.clamps.append(clamp)

    def optimize(self) -> None:
        """TODO"""
