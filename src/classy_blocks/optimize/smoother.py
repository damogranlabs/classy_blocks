from typing import List

import numpy as np

from classy_blocks.mesh import Mesh
from classy_blocks.optimize.grid import Grid
from classy_blocks.optimize.junction import Junction


class SmootherBase:
    def __init__(self, grid: Grid):
        self.grid = grid

        self.inner: List[Junction] = []
        for junction in self.grid.junctions:
            if not junction.is_boundary:
                self.inner.append(junction)

    def smooth_iteration(self) -> None:
        for junction in self.inner:
            near_points = [j.point for j in junction.neighbours]
            junction.move_to(np.average(near_points, axis=0))

    def smooth(self, iterations: int = 5) -> None:
        for _ in range(iterations):
            self.smooth_iteration()


class MeshSmoother(SmootherBase):
    def __init__(self, mesh: Mesh):
        self.mesh = mesh

        super().__init__(Grid.from_mesh(self.mesh))

    def smooth(self, iterations: int = 5) -> None:
        super().smooth(iterations)

        for i, point in enumerate(self.grid.points):
            self.mesh.vertices[i].move_to(point)
