from classy_blocks.mesh import Mesh
from classy_blocks.modify.grid import Grid


class Optimizer:
    """Provides tools for blocking optimization"""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

        self.grid = Grid(mesh)
