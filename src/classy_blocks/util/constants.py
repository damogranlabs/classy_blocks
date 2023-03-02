import numpy as np

# data type
DTYPE = 'float' # dtype as taken by np.array()

# geometric tolerance
TOL = 1e-7


# Block definition:
# a more intuitive and quicker way to set patches,
# according to this sketch: https://www.openfoam.com/documentation/user-guide/blockMesh.php
# the same for all blocks
FACE_MAP = {
    'bottom': (0, 1, 2, 3),
    'top': (4, 5, 6, 7),
    'left': (4, 0, 3, 7),
    'right': (5, 1, 2, 6),
    'front': (4, 5, 1, 0),
    'back': (7, 6, 2, 3),
}

# pairs of corner indexes along axes
AXIS_PAIRS = (
    ((0, 1), (3, 2), (7, 6), (4, 5)),  # x
    ((0, 3), (1, 2), (5, 6), (4, 7)),  # y
    ((0, 4), (1, 5), (2, 6), (3, 7)),  # z
)

# pairs of corner indexes that define edges (and not diagonals)
EDGE_PAIRS = (
    list(AXIS_PAIRS[0]) + \
    list(AXIS_PAIRS[1]) + \
    list(AXIS_PAIRS[2])
)

# number formatting
def vector_format(vector) -> str:
    # output for edge definitions
    return f"({vector[0]:.8f} {vector[1]:.8f} {vector[2]:.8f})"

# Circle H-grid parameters
# TODO: move this to Shape or something
# A quarter of a circle is created from 3 blocks;
# Central 'square' (0) and two curved 'rectangles' (1 and 2)
#
# |*******
# |  2    /**
# |      /    *
# S-----D      *
# |  0  |   1   *
# |_____S_______*
# O
#
# Relative size of the inner square (O-D):
# too small will cause unnecessary high number of small cells in the square;
# too large will prevent creating large numbers of boundary layers
circle_core_diagonal = 0.7
# Orthogonality of the inner square (O-S):
circle_core_side = 0.62

# Sphere parameters:
# The same sketch as above but it represents sphere cross-section.
# Vector O-S is circle normal; there are two different angles DOS;
sphere_diagonal_angle = np.pi / 4
sphere_side_angle = np.pi / 6

MESH_HEADER =  ("/*---------------------------------------------------------------------------*\\\n"
                "| =========                 |                                                 |\n"
                "| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n"
                "|  \\    /   O peration     | Version:  v1806/v10                             |\n"
                "|   \\  /    A nd           | Web:      https://www.OpenFOAM.com              |\n"
                "|    \\/     M anipulation  |           https://www.OpenFOAM.org              |\n"
                "\\*---------------------------------------------------------------------------*/\n"
                "FoamFile\n"
                "{\n"
                "    version     2.0;\n"
                "    format      ascii;\n"
                "    class       dictionary;\n"
                "    object      blockMeshDict;\n"
                "}\n"
                "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n")

MESH_FOOTER = (
    "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
    "// Created with classy_blocks: https://github.com/damogranlabs/classy_blocks //\n"
    "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
)
