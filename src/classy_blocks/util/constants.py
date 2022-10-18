import numpy as np

# geometric tolerance
tol = 1e-7


# number formatting
def vector_format(vector):
    # output for edge definitions
    return f"({vector[0]:.8f} {vector[1]:.8f} {vector[2]:.8f})"


# Circle H-grid parameters
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
