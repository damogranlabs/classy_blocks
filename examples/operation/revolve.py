import os

from classes.primitives import Edge
from classes.mesh import Mesh

from operations.base import Face
from operations.operations import Revolve

from util.methematics import functions as g

def create():
    base = Face(
        [ [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0] ],
        [ [0.5, -0.2, 0], None, None, None]
    )

    revolve = Revolve(base, g.deg2rad(60), [0, -1, 0], [-2, 0, 0])

    revolve.set_cell_count(0, 15)
    revolve.set_cell_count(1, 10)
    revolve.set_cell_count(2, 50)

    revolve.grade_to_size(2, -0.02)

    mesh = Mesh()
    mesh.add_operation(revolve)

    mesh.write('util/blockMeshDict.template', 'examples/meshCase/system/blockMeshDict')

    # run blockMesh
    os.system("blockMesh -case examples/meshCase")