import os
import random

import classy_blocks as cb
from classy_blocks.construct.flat.sketches.spline_round import SplineDisk

mesh = cb.Mesh()

height = 2
edge_sketches = 4

sketch = SplineDisk([0, 0, 0], [2, 0, 0], [0, 1, 0], 1, 0.3)

dz = height / (edge_sketches + 1)
xs_sketches = [sketch.copy().translate([random.random() / 10, random.random() / 10, dz])]

for _ in range(edge_sketches - 1):
    xs_sketches.append(xs_sketches[-1].copy().translate([random.random() / 10, random.random() / 10, dz]))

extrude = cb.LoftedShape(sketch, sketch.copy().translate([0, 0, height]), xs_sketches)

for axis in range(3):
    extrude.chop(axis, count=10)


mesh.add(extrude)
mesh.write(os.path.join("..", "..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
