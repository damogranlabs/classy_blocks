import os

import classy_blocks as cb
from classy_blocks.construct.flat.sketches.spline_round import QuarterSplineRoundRing

mesh = cb.Mesh()


sketch = QuarterSplineRoundRing([0, 0, 0], [1, 0, 0], [0, 1.2, 0], 0.2, 0.3, 0.1, 0.2)
ring = cb.ExtrudedShape(sketch, 1)

for operation in ring.operations:
    for i in range(3):
        operation.chop(i, count=10)

mesh.add(ring)
mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
