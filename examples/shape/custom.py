import os

import numpy as np

import classy_blocks as cb
from classy_blocks.grading.autograding.grader import SmoothGrader
from classy_blocks.types import PointType
from classy_blocks.util import functions as f

# an example with a custom sketch, yielding a custom shape (square with rounded corners);
# see custom.svg for a sketch of blocking scheme
mesh = cb.Mesh()


class RoundSquare(cb.MappedSketch):
    quads = [
        [20, 0, 1, 12],
        [12, 1, 2, 13],
        [13, 2, 3, 21],
        [21, 20, 12, 13],
        [21, 3, 4, 14],
        [14, 4, 5, 15],
        [15, 5, 6, 22],
        [22, 21, 14, 15],
        [22, 6, 7, 16],
        [16, 7, 8, 17],
        [17, 8, 9, 23],
        [23, 22, 16, 17],
        [23, 9, 10, 18],
        [18, 10, 11, 19],
        [19, 11, 0, 20],
        [20, 23, 18, 19],
        [21, 22, 23, 20],
    ]

    def __init__(self, center: PointType, side: float, corner_round: float):
        center = np.array(center)
        points = [
            center + f.vector(side / 2, 0, 0),
            center + f.vector(side / 2, side / 2 - side * corner_round / 2, 0),
            center + f.vector(side / 2 - side * corner_round / 2, side / 2, 0),
        ]

        outer_points = []

        angles = np.linspace(0, 2 * np.pi, num=4, endpoint=False)
        for a in angles:
            for i in range(3):
                outer_points.append(f.rotate(points[i], a, [0, 0, 1], center))

        inner_points = np.ones((12, 3)) * center
        super().__init__(np.concatenate((outer_points, inner_points), axis=0), RoundSquare.quads)

    def add_edges(self) -> None:
        for i in (1, 5, 9, 13):
            self.faces[i].add_edge(1, cb.Angle(np.pi / 2, [0, 0, 1]))


base_1 = RoundSquare([0, 0, 0], 1, 0.5)
smoother = cb.SketchSmoother(base_1)
smoother.smooth()

shape = cb.ExtrudedShape(base_1, 1)

mesh.add(shape)
mesh.assemble()

grader = SmoothGrader(mesh, 0.03)
grader.grade(take="max")

mesh.set_default_patch("walls", "wall")
mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
