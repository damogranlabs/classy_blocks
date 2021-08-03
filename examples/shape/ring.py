from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.shapes import ExtrudedRing

mesh = Mesh()

pipe_wall = ExtrudedRing(
    [0, 0, 0], [2, 2, 0], [-0.707, 0.707, 0], 1.2
)

pipe_wall.set_axial_cell_count(20)
pipe_wall.set_radial_cell_count(10)
pipe_wall.set_tangential_cell_count(20)

pipe_wall.set_bottom_patch('inlet')
pipe_wall.set_top_patch('outlet')
pipe_wall.set_inner_patch('inner_wall')
pipe_wall.set_outer_patch('outer_wall')

pipe_wall.set_outer_cell_size(-0.01)

mesh.add_shape(pipe_wall)

mesh.write('case/system/blockMeshDict')
