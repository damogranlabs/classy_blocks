#!/usr/bin/env python
import os

from classy_blocks.classes.operations import Face, Extrude
from classy_blocks.classes.mesh import Mesh

base = Face(
    [ [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0] ],
    [ [0.5, -0.2, 0], None, None, None]
)

extrude = Extrude(base, [0.5, 0.5, 3])
extrude.set_cell_count(0, 15)
extrude.set_cell_count(1, 10)
extrude.set_cell_count(2, 50)

extrude.grade_to_size(2, -0.02)

mesh = Mesh()
mesh.add_operation(extrude)

mesh.write('case/system/blockMeshDict')