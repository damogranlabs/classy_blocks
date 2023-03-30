import os
import classy_blocks as cb

mesh = cb.Mesh()

# points that define ring cross-section;
# must be specified in the following order:
#         p3---___
#        /        ---p2
#       /              \
#      p0---------------p1
#
# 0---- -- ----- -- ----- -- ----- -- --->> axis
xs_points = [
    [0.1, 0.2, 0],
    [0.8, 0.1, 0],
    [0.7, 0.5, 0],
    [0.2, 0.5, 0],
]

xs_edges = [None, None, cb.Arc([0.3, 0.55, 0]), None]  # these must be consistent with points

face = cb.Face(xs_points, xs_edges)

pipe_wall = cb.RevolvedRing([0, 0, 0], [1, 0, 0], face)  # axis_point_1  # axis_point_2,

core_size = 0.05
pipe_wall.chop_axial(start_size=core_size)
pipe_wall.chop_tangential(start_size=core_size)
pipe_wall.chop_radial(count=10)

pipe_wall.set_start_patch("inlet")
pipe_wall.set_end_patch("outlet")
pipe_wall.set_inner_patch("inner_wall")
pipe_wall.set_outer_patch("outer_wall")

mesh.add(pipe_wall)

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
