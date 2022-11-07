from classy_blocks.process.mesh import Mesh
from classy_blocks.construct.shapes import ExtrudedRing

def get_mesh():
    mesh = Mesh()

    pipe_wall = ExtrudedRing(
        [0, 0, 0], # axis_point_1
        [2, 2, 0], # axis_point_2, 
        [-0.707, 0.707, 0], # outer_radius
        0.8, # inner_radius
        n_segments=12 # not required, 4 is enough
    )

    core_size = 0.1
    bl_thickness = 0.005

    pipe_wall.chop_axial(start_size=core_size)
    pipe_wall.chop_tangential(start_size=core_size)
    
    # chop radially twice to get boundary layer cells on both sides;
    pipe_wall.chop_radial(length_ratio=0.5, start_size=bl_thickness, c2c_expansion=1.2)
    pipe_wall.chop_radial(length_ratio=0.5, end_size=bl_thickness, c2c_expansion=1/1.2)

    pipe_wall.set_bottom_patch('inlet')
    pipe_wall.set_top_patch('outlet')
    pipe_wall.set_inner_patch('inner_wall')
    pipe_wall.set_outer_patch('outer_wall')

    mesh.add(pipe_wall)

    return mesh
