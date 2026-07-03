"""A conical frustum but with a square top cross-section"""

import os

import classy_blocks as cb
from classy_blocks.cbtyping import PointType
from classy_blocks.construct.flat.sketches.disk import FourCoreDisk, VectorType
from classy_blocks.util import functions as f

mesh = cb.Mesh()

axis_point_1 = f.vector(0, 0, 0)
axis_point_2 = f.vector(2, 0, 0)
radius_point_1 = f.vector(0, 0, 2)
radius_point_2 = f.vector(2, 0, 1)

bl_thickness = 0.01
core_size = 0.1


class FourCoreSquare(FourCoreDisk):
    def __init__(self, center_point: PointType, radius_point: PointType, normal: VectorType):
        super().__init__(center_point, radius_point, normal)

        # correct points 10, 12, 14, 16
        # "factor" 1.4 is sqrt(2);
        # factor 1.2 is somewhat less (defined empirically)
        pos = self.positions
        for i in (10, 12, 14, 16):
            radius_vector = pos[i] - pos[0]
            pos[i] = pos[0] + 2**0.5 * radius_vector

            # also correct points 2, 4, 6, 8
            inner_radius_vector = pos[i - 8] - pos[0]
            pos[i - 8] = pos[0] + 1.2 * inner_radius_vector

        self.update(pos)

        # remove all curved edges
        for face in self.faces:
            face.remove_edges()


translation = axis_point_2 - axis_point_1
normal = f.unit_vector(translation)

bottom_sketch = FourCoreDisk(axis_point_1, radius_point_1, normal)
top_sketch = FourCoreSquare(axis_point_2, radius_point_2, normal)

frustum = cb.LoftedShape(bottom_sketch, top_sketch)

frustum.chop(0, start_size=core_size, end_size=bl_thickness)
frustum.chop(1, start_size=core_size)
frustum.chop(2, count=30)

mesh.add(frustum)
mesh.set_default_patch("walls", "wall")
mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
