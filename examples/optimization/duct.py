# An example where a Shape is optimized *before* it is added to mesh, using ShapeOptimizer
import os

import classy_blocks as cb
from classy_blocks.optimize.junction import ClampExistsError
from classy_blocks.optimize.optimizer import ShapeOptimizer

mesh = cb.Mesh()

start_sketch = cb.SplineDisk([0, 0, 0], [2.5, 0, 0], [0, 1, 0], 0, 0)
end_sketch = cb.SplineDisk([0, 0, 0], [1, 0, 0], [0, 2.5, 0], 0, 0).translate([0, 0, 1])

shape = cb.LoftedShape(start_sketch, end_sketch)

optimizer = ShapeOptimizer(shape.operations)

for operation in shape.operations[:4]:
    # remove edges because inner splines will ruin the day
    # TODO: make edges move with points too
    operation.top_face.remove_edges()
    operation.bottom_face.remove_edges()

    for point in operation.point_array:
        try:
            optimizer.add_clamp(cb.FreeClamp(point))
        except ClampExistsError:
            pass

optimizer.optimize(tolerance=0.01)

# Quick'n'dirty chopping, don't do this at home
for operation in shape.operations:
    for axis in range(3):
        operation.chop(axis, count=10)

mesh.add(shape)

mesh.set_default_patch("walls", "wall")
mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
