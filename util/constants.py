# geometric tolerance
tol = 1e-7

# number formatting
def vector_format(vector):
    # output for edge definitions
    return "({:.8f} {:.8f} {:.8f})".format(
        vector[0],
        vector[1],
        vector[2]
    )

# cylinder/frustum creation:
# ratio of radius of internal block vertex of a cylinder/frustum
# to outer radius of that cylinder/frustum
frustum_core_to_outer = 0.4
# the same as above but is used for edge points
frustum_edge_to_outer = 0.31