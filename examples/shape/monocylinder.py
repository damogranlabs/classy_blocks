import os

import numpy as np

import classy_blocks as cb
from classy_blocks.construct.flat.sketches.disk import QuarterDisk
from classy_blocks.construct.flat.sketches.mapped import MappedSketch
from classy_blocks.construct.shapes.shape import LoftedShape
from classy_blocks.util import functions as f

# An example case with a Shape, created from a custom sketch.
# In this case, a cylinder with a single block in the middle and four blocks
# surrounding it.
mesh = cb.Mesh()

center_point = np.array([0, 0, 0])
radius_point = np.array([1, 0, 0])


class MonoCylinderSketch(MappedSketch):
    def add_edges(self):
        for i in (1, 2, 3, 4):
            self.faces[i].add_edge(1, cb.Origin(center_point))


# a cylinder with a single block in the center
angles = np.linspace(0, 2 * np.pi, num=4, endpoint=False)

inner_points = [f.rotate(radius_point * QuarterDisk.diagonal_ratio, a, [0, 0, 1], center_point) for a in angles]
outer_points = [f.rotate(radius_point, a, [0, 0, 1], center_point) for a in angles]

locations = inner_points + outer_points
quads = [
    (0, 1, 2, 3),
    (0, 4, 5, 1),
    (1, 5, 6, 2),
    (2, 6, 7, 3),
    (3, 7, 4, 0),
]

base = MonoCylinderSketch(locations, quads, 5)

cylinder = LoftedShape(base, base.copy().translate([0, 0, 1]))
cylinder.operations[1].chop(2, count=10)

cylinder.operations[1].chop(1, count=5)
cylinder.operations[2].chop(1, count=5)

cylinder.operations[1].chop(0, count=3)

mesh.add(cylinder)
mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
