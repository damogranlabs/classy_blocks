from typing import List

import numpy as np

from classy_blocks.construct.flat.sketches.mapped import MappedSketch
from classy_blocks.mesh import Mesh
from classy_blocks.optimize.grid import GridBase, HexGrid, QuadGrid
from classy_blocks.optimize.junction import Junction


class SmootherBase:
    def __init__(self, grid: GridBase):
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

        super().__init__(HexGrid.from_mesh(self.mesh))

    def smooth(self, iterations: int = 5) -> None:
        super().smooth(iterations)

        for i, point in enumerate(self.grid.points):
            self.mesh.vertices[i].move_to(point)


class SketchSmoother(SmootherBase):
    def __init__(self, sketch: MappedSketch):
        self.sketch = sketch

        grid = QuadGrid.from_sketch(self.sketch)

        super().__init__(grid)

    def smooth(self, iterations: int = 5) -> None:
        super().smooth(iterations)

        positions = self.grid.points

        for i, quad in enumerate(self.sketch.indexes):
            points = np.take(positions, quad, axis=0)

            self.sketch.faces[i].update(points)
