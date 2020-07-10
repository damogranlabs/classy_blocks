import os

from classes.primitives import Edge
from classes.mesh import Mesh

from operations.base import Face
from operations.operations import Extrude

def create():
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

    mesh.write('util/blockMeshDict.template', 'examples/meshCase/system/blockMeshDict')

    # run blockMesh
    os.system("blockMesh -case examples/meshCase")