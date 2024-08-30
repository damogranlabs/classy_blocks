import os

import numpy as np

import classy_blocks as cb
from classy_blocks.util import functions as f

# This rather lengthy tutorial does the following:
# - reads 2D airfoil points*
# - rotates the airfoil to a desired angle of attack
# - scales it to actual chord length
# - creates a small domain around the airfoil
# - maintains thickness of 1 in 3rd dimension**
# - optimizes its blocking for best results

# The example does not aim to produce production-ready
# airfoil meshes but serves as a demonstration of:
# - creation and manipulation of curves
# - usage of OnCurve edges
# - optimization of 2D geometry with translation links on points in 3rd dimension

# * A word on trailing edges
# NACA equations produce a blunt trailing edge if no correction is applied
# and this 'feature' is exploited here so that a blocking where
# thin boundary layer cells do not spread into the domain downstream.
# This blocking will not work with sharp trailing edges.
# Also, real-life geometry will not work with infinitely
# sharp trailing edges (or people around them won't).

# ** This is a default for 2D OpenFOAM cases.
# It will make postprocessing easier but in cases where
# chords are on a much different scale (say, 0.1 or 10),
# optimization will probably have difficulties.
# That is because it tries to keep aspect ratios of blocks
# as near to 1 as possible but for 2D cases it doesn't matter.


FILE_NAME = "naca2414.dat"
ANGLE_OF_ATTACK = 20  # in degrees
CHORD = 0.5  # desired chord (provided the one from points is 1)
OPTIMIZE = True  # Set to False to skip optimization

CELL_SIZE = 0.025
BL_THICKNESS = 0.001  # thickness of boundary layer cells
C2C_EXPANSION = 1.1  # expansion ratio in boundary layer

thickness = f.vector(0, 0, 1)  # a shortcut


### Load curves
# one for bottom faces, one for top faces;
# Optimization will refer to the bottom curve, top points
# simply follow their 'leader' points.
def get_curve(z: float) -> cb.SplineInterpolatedCurve:
    """Loads 2D points from a Selig file and
    converts it to 3D by adding a provided z-coordinate."""
    raw_data = np.loadtxt(FILE_NAME, skiprows=1) * CHORD
    z_dimensions = np.ones((len(raw_data),)) * z
    points = np.hstack((raw_data, z_dimensions[:, None]))

    curve = cb.SplineInterpolatedCurve(points)
    curve.rotate(f.deg2rad(-ANGLE_OF_ATTACK), [0, 0, 1])

    return curve


foil_curve = get_curve(0)
top_foil_curve = get_curve(1)

### Select approximate point positions:
# refer to the airfoil.svg sketch for explanation
# and indexes
points = np.zeros((18, 3))

points[0] = [-CHORD / 2, 0, 0]
points[1] = [0, CHORD / 2, 0]
points[2] = [CHORD, CHORD / 2, 0]
points[3] = [1.5 * CHORD, CHORD / 2, 0]
points[4] = [1.5 * CHORD, CHORD / 4, 0]
points[5] = [1.5 * CHORD, -CHORD / 4, 0]
points[6] = [1.5 * CHORD, -CHORD / 2, 0]
points[7] = [CHORD, -CHORD / 2, 0]
points[8] = [0, -CHORD / 2, 0]

for i, point in enumerate(foil_curve.discretize(count=7)):
    # points 9...15
    points[i + 9] = point

points[12] = foil_curve.get_point(foil_curve.get_closest_param(points[0]))

points[16] = np.average(np.take(points, (9, 4, 3, 2), axis=0), axis=0)
points[17] = np.average(np.take(points, (15, 5, 6, 7), axis=0), axis=0)


### Create lofts:
def get_loft(indexes):
    bottom_face = cb.Face(np.take(points, indexes, axis=0))
    top_face = bottom_face.copy().translate(thickness)

    loft = cb.Loft(bottom_face, top_face)
    loft.set_patch(["top", "bottom"], "topAndBottom")

    return loft


mesh = cb.Mesh()

loft_indexes = [
    [12, 11, 1, 0],  # 0
    [11, 10, 2, 1],  # 1
    [10, 9, 16, 2],  # 2
    [9, 15, 17, 16],  # 3
    [15, 14, 7, 17],  # 4
    [14, 13, 8, 7],  # 5
    [13, 12, 0, 8],  # 6
    [2, 16, 4, 3],  # 7
    [16, 17, 5, 4],  # 8
    [17, 7, 6, 5],  # 9
]

lofts = [get_loft(quad) for quad in loft_indexes]

# Create curved edges
for i in (0, 1, 2, 4, 5, 6):
    loft = lofts[i]
    loft.bottom_face.add_edge(0, cb.OnCurve(foil_curve))
    loft.top_face.add_edge(0, cb.OnCurve(top_foil_curve))

# Round edges of blocks 0 and 6
for i in (0, 6):
    lofts[i].bottom_face.add_edge(2, cb.Origin([0, 0, 0]))
    lofts[i].top_face.add_edge(2, cb.Origin(thickness))

### Set cell size/grading;
# Keep in mind that not all blocks need exact specification as
# chopping will propagate automatically through blocking
lofts[0].chop(2, count=1)  # 1 cell in the 3rd dimension
# keep consistent first cell thickness by using edge grading
lofts[1].chop(1, start_size=BL_THICKNESS, c2c_expansion=C2C_EXPANSION, preserve="start_size")
lofts[8].chop(1, start_size=CELL_SIZE)

for i in (0, 1, 2, 3, 4, 5, 6):
    lofts[i].chop(0, start_size=CELL_SIZE)

for loft in lofts:
    mesh.add(loft)

### Optimize:
mesh.assemble()
finder = cb.GeometricFinder(mesh)
optimizer = cb.MeshOptimizer(mesh)


# Some helper functions
def find_vertex(index):
    return list(finder.find_in_sphere(points[index]))[0]


def make_link(leader):
    # Find the respective point in the top plane and
    # create a link so that it will follow when
    # leader's position changes
    follower_position = leader.position + thickness
    follower = list(finder.find_in_sphere(follower_position))[0]

    return cb.TranslationLink(leader.position, follower.position)


# Points that slide along airfoil curve
for index in (10, 11, 12, 13, 14):
    opt_vertex = find_vertex(index)
    clamp = cb.CurveClamp(opt_vertex.position, foil_curve)
    optimizer.add_link(make_link(opt_vertex))

    optimizer.add_clamp(clamp)

# Points that move in their X-Y plane
for index in (16, 17):
    opt_vertex = find_vertex(index)
    clamp = cb.PlaneClamp(opt_vertex.position, [0, 0, 0], [0, 0, 1])
    optimizer.add_link(make_link(opt_vertex))
    optimizer.add_clamp(clamp)


# Points that move along domain edges
def optimize_along_line(point_index, line_index_1, line_index_2):
    opt_vertex = find_vertex(point_index)
    clamp = cb.LineClamp(opt_vertex.position, points[line_index_1], points[line_index_2])
    optimizer.add_link(make_link(opt_vertex))
    optimizer.add_clamp(clamp)


optimize_along_line(2, 1, 3)
optimize_along_line(7, 8, 6)
optimize_along_line(4, 3, 6)
optimize_along_line(5, 3, 6)

if OPTIMIZE:
    optimizer.optimize(tolerance=1e-3, method="SLSQP")
    mesh.backport()
    mesh.clear()

### Write the mesh
mesh.modify_patch("topAndBottom", "empty")
mesh.set_default_patch("freestream", "patch")
mesh.write(os.path.join("..", "..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
