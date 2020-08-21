import os

import numpy as np

from classes.primitives import Edge
from classes.mesh import Mesh

from operations.base import Face
from operations.operations import Wedge

from util.methematics import functions as g

def create():
    base = Face(
        [ [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0] ],
        [ None, None, [
            [0.75, 0.9, 0], # a spline edge
            [0.50, 1.0, 0], # with 3
            [0.25, 1.1, 0], # points
        ],
        None]
    )

    mesh = Mesh()

    wedges = []
    for _ in range(5):
        wedge = Wedge(base)

        wedge.set_cell_count(0, 15)
        wedge.set_cell_count(1, 30)
        
        wedge.grade_to_size(1, -0.01)
        wedge.set_outer_patch('wall')
        wedges.append(wedge)

        mesh.add_operation(wedge)

        base = base.translate([1, 0, 0])

    wedges[0].set_left_patch('inlet')
    wedges[-1].set_right_patch('outlet')

    mesh.write('util/blockMeshDict.template', 'examples/meshCase/system/blockMeshDict')

    # run blockMesh
    os.system("blockMesh -case examples/meshCase")