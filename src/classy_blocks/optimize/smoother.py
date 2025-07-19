import abc
from collections.abc import Iterable

import numpy as np

from classy_blocks.cbtyping import PointListType
from classy_blocks.construct.flat.sketches.mapped import MappedSketch
from classy_blocks.mesh import Mesh
from classy_blocks.optimize.grid import GridBase, HexGrid, QuadGrid
from classy_blocks.optimize.junction import Junction
from classy_blocks.util import functions as f
from classy_blocks.util.constants import TOL


class SmootherBase(abc.ABC):
    def __init__(self, grid: GridBase):
        self.grid = grid

        self.inner: list[Junction] = []
        for junction in self.grid.junctions:
            if not junction.is_boundary:
                self.inner.append(junction)

        self.fixed: set[int] = set()

    def fix_indexes(self, indexes: Iterable[int]) -> None:
        self.fixed.update(set(indexes))

    def fix_points(self, points: PointListType):
        for point in points:
            for junction in self.grid.junctions:
                if f.norm(point - junction.point) < TOL:
                    self.fixed.add(junction.index)

    def smooth(self, iterations: int = 5) -> None:
        for _ in range(iterations):
            for junction in self.inner:
                if junction.index in self.fixed:
                    continue

                near_points = [j.point for j in junction.neighbours]
                self.grid.points[junction.index] = np.average(near_points, axis=0)

        self.backport()

    @abc.abstractmethod
    def backport(self) -> None:
        """Copy results of smoothing back to the grid"""


class MeshSmoother(SmootherBase):
    def __init__(self, mesh: Mesh):
        self.mesh = mesh

        super().__init__(HexGrid.from_mesh(self.mesh))

    def backport(self):
        for i, point in enumerate(self.grid.points):
            self.mesh.vertices[i].move_to(point)


class SketchSmoother(SmootherBase):
    def __init__(self, sketch: MappedSketch):
        self.sketch = sketch

        grid = QuadGrid.from_sketch(self.sketch)

        super().__init__(grid)

    def backport(self):
        positions = self.grid.points

        for i, quad in enumerate(self.sketch.indexes):
            points = np.take(positions, quad, axis=0)

            self.sketch.faces[i].update(points)
