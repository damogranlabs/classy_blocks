from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.shapes import Cylinder, ExtrudedRing

# not finished!

pipe_internal_diameter = 20e-3 # [m]
pipe_wall_thickness = 2e-3 # [m]
block_length = 0.2 # [m]

pipe_spacing = 0.1

n_cells_axial = 30
cell_size = block_length / n_cells_axial

mesh = Mesh()


def exchanger_block(x, y):
    # pipe: inside mesh
    diagonal = 0.25*pipe_internal_diameter * 2**0.5

    pipe_inside = Cylinder(
        [x, y, 0],
        [x, y, block_length],
        [x + diagonal, y + diagonal, 0])

    pipe_inside.set_axial_cell_count(n_cells_axial)
    pipe_inside.set_tangential_cell_count(10)
    pipe_inside.set_radial_cell_count(10)

    pipe_inside.set_cell_zone('pipe_inside')

    mesh.add_shape(pipe_inside)

    # pipe: wall mesh
    pipe_wall = ExtrudedRing(
        [x, y, 0],
        [x, y, block_length],
        [x + diagonal, y + diagonal, 0],
        0.5*pipe_internal_diameter + pipe_wall_thickness
    )

    pipe_wall.set_radial_cell_count(5)
    pipe_wall.set_cell_zone('pipe_wall')

    mesh.add_shape(pipe_wall)

exchanger_block(0, 0)

mesh.write('case/system/blockMeshDict')
