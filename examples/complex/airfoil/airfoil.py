from typing import ClassVar

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
# - Sketch optimization (2D geometry)

# * A word on trailing edges
# NACA equations produce a blunt trailing edge if no correction is applied
# and this 'feature' is exploited here so that a blocking where
# thin boundary layer cells do not spread into the domain downstream.
# This blocking will not work with sharp trailing edges.
# Also, real-life geometry will not work with infinitely
# sharp trailing edges (or people around them won't).

# ** This is a default for 2D OpenFOAM cases


FILE_NAME = "naca2414.dat"
ANGLE_OF_ATTACK = 10  # in degrees
CHORD = 0.5  # desired chord (provided the one from points is 1)
OPTIMIZE = True  # Set to False to skip optimization

CELL_SIZE = 0.025
BL_THICKNESS = 0.001  # thickness of boundary layer cells
C2C_EXPANSION = 1.1  # expansion ratio in boundary layer

mesh = cb.Mesh()


### Load airfoil curve
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

# points 9...15 with point 12 being set first
param_12 = foil_curve.get_closest_param(points[0])
curve_params = np.concatenate(
    (np.linspace(0, param_12, num=3, endpoint=False), [param_12], np.linspace(param_12, 1, num=4)[1:])
)

for i, t in enumerate(curve_params):
    points[i + 9] = foil_curve.get_point(t)


# create a sketch on the bottom
class AirfoilSketch(cb.MappedSketch):
    quads: ClassVar = [
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

    def __init__(self, points):
        super().__init__(points, self.quads)

        # determine initial positions for points 16 and 17
        # with smoothing
        smoother = cb.SketchSmoother(self)
        smoother.smooth()

        # Optimize:
        optimizer = cb.SketchOptimizer(self)
        pos = self.positions

        # Points that slide along the airfoil curve
        for i, point_index in enumerate((10, 11, 12, 13, 14)):
            initial_param = curve_params[i]
            optimizer.add_clamp(cb.CurveClamp(pos[point_index], foil_curve, initial_param))

        # Points that move in their X-Y plane
        for i in (16, 17):
            optimizer.add_clamp(cb.PlaneClamp(pos[i], pos[i], [0, 0, 1]))

        # Points that move along domain edges
        def optimize_along_line(point_index, line_index_1, line_index_2):
            clamp = cb.LineClamp(pos[point_index], pos[line_index_1], pos[line_index_2])
            optimizer.add_clamp(clamp)

        optimize_along_line(2, 1, 3)
        optimize_along_line(7, 8, 6)
        optimize_along_line(4, 3, 6)
        optimize_along_line(5, 3, 6)

        # point 0: on arc
        clamp = cb.RadialClamp(pos[0], [0, 0, 0], [0, 0, 1])
        optimizer.add_clamp(clamp)

        optimizer.optimize(tolerance=1e-5, method="SLSQP", relax=True)

        # edges were added in super().__init__() but now positions have changed and
        # we have to adjust for that
        self.add_edges()

    def add_edges(self):
        for i in (0, 1, 2, 4, 5, 6):
            self.faces[i].add_edge(0, cb.OnCurve(foil_curve.copy()))

        for i in (0, 6):
            self.faces[i].add_edge(2, cb.Origin([0, 0, 0]))


### Create an extruded shape
base = AirfoilSketch(points)
shape = cb.ExtrudedShape(base, [0, 0, 1])

### Set cell size/grading;
# Keep in mind that not all blocks need exact specification as
# chopping will propagate automatically through blocking
shape.chop(2, count=1)  # 1 cell in the 3rd dimension (2D domain)
# keep consistent first cell thickness by using edge grading
shape.operations[1].chop(1, start_size=BL_THICKNESS, c2c_expansion=C2C_EXPANSION, take="max", preserve="start_size")
# This is guesswork! Can be solved with a different blocking (that will product more even block sizes),
# a lot of math (check the cell size of block 3) or an automatic grader (TODO).
shape.operations[8].chop(1, start_size=BL_THICKNESS, end_size=CELL_SIZE, preserve="end_size")

### Set patches
shape.set_start_patch("topAndBottom")
shape.set_end_patch("topAndBottom")

for i in range(7):
    shape.operations[i].set_patch("front", "airfoil")

mesh.modify_patch("airfoil", "wall")
mesh.modify_patch("topAndBottom", "empty")
mesh.set_default_patch("freestream", "patch")

mesh.add(shape)

# Chop remaining blocks with an automatic grader (see the comment at manual grading above)
grader = cb.SimpleGrader(mesh, CELL_SIZE)
grader.grade(take="max")

### Write the mesh
mesh.write("../../case/system/blockMeshDict", debug_path="debug.vtk")
