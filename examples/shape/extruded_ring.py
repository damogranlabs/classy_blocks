import os

import classy_blocks as cb

mesh = cb.Mesh()

pipe_wall = cb.ExtrudedRing(
    [0, 0, 0], [2, 2, 0], [-0.707, 0.707, 0], 0.8  # axis_point_1  # axis_point_2  # outer_radius  # inner_radius
)

core_size = 0.1
bl_thickness = 0.005

pipe_wall.chop_axial(start_size=core_size)
pipe_wall.chop_tangential(start_size=core_size)

# chop radially twice to get boundary layer cells on both sides;
pipe_wall.chop_radial(length_ratio=0.5, start_size=bl_thickness, c2c_expansion=1.2)
pipe_wall.chop_radial(length_ratio=0.5, end_size=bl_thickness, c2c_expansion=1 / 1.2)

pipe_wall.set_start_patch("inlet")
pipe_wall.set_end_patch("outlet")
pipe_wall.set_inner_patch("inner_wall")
pipe_wall.set_outer_patch("outer_wall")

mesh.add(pipe_wall)

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
