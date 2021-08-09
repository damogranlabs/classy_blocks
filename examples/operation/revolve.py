import os

from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.operations import Face, Revolve

from classy_blocks.util import functions as f

base = Face(
    [ [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0] ],
    [ [0.5, -0.2, 0], None, None, None]
)

revolve = Revolve(base, f.deg2rad(60), [0, -1, 0], [-2, 0, 0])

revolve.set_cell_count(0, 15)
revolve.set_cell_count(1, 10)
revolve.set_cell_count(2, 50)

revolve.grade_to_size(2, -0.02)

mesh = Mesh()
mesh.add(revolve)

mesh.write('case/system/blockMeshDict')
