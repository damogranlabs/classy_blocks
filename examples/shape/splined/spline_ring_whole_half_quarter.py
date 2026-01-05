import os

import classy_blocks as cb

mesh = cb.Mesh()

quarter_ring_sketch = cb.QuarterSplineRing([0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], 0.1, 0.2, 0.8, 0.8)
half_ring_sketch = cb.HalfSplineRing([0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], 0.1, 0.2, 0.8, 0.8)
whole_ring_sketch = cb.SplineRing([0, 0, 0], [1.0, 0, 0], [0, 1.0, 0], 0.1, 0.2, 0.8, 0.8)

sketches = [quarter_ring_sketch, half_ring_sketch, whole_ring_sketch]

extruded_shapes = []

for i, sketch in enumerate(sketches):
    extruded_shapes.append(cb.ExtrudedShape(sketch, 1))
    extruded_shapes[i].translate([0, 0, i * 3])

    for operation in extruded_shapes[i].operations:
        for j in range(3):
            operation.chop(j, count=10)

    mesh.add(extruded_shapes[i])

mesh.write(os.path.join("..", "..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
