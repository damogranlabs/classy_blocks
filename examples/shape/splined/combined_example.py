import os

import numpy as np

import classy_blocks as cb

center_point = np.asarray([0, 0, 0])
direction = np.asarray([1, 0, 0])

# Sketches can be completely circular. Here Arc are used to define the outer edges.
circular_sketch_1 = cb.SplineDisk(
    center_point=center_point,
    corner_1_point=np.asarray([0, 1, 0]),
    corner_2_point=np.asarray([0, 0, 1]),
    side_1=0,
    side_2=0,
)
# Sketches can be elliptical. The outer edges are defined by splines.
# For high accuracy more spline points can be added using the key word n_outer_spline_points, in the initialization.
elliptical_sketch_1 = cb.SplineDisk(
    center_point=center_point + direction * 1,
    corner_1_point=np.asarray([0, 1, 0]) + direction * 1,
    corner_2_point=np.asarray([0, 0, 1.2]) + direction * 1,
    side_1=0,
    side_2=0,
    n_outer_spline_points=50,
)

elliptical_sketch_2 = cb.SplineDisk(
    center_point=center_point + direction * 2,
    corner_1_point=np.asarray([0, 1, 0]) + direction * 2,
    corner_2_point=np.asarray([0, 0, 1.5]) + direction * 2,
    side_1=0,
    side_2=0,
)

elliptical_sketch_3 = cb.SplineDisk(
    center_point=center_point + direction * 3,
    corner_1_point=np.asarray([0, 1.1, 0]) + direction * 3,
    corner_2_point=np.asarray([0, 0, 2]) + direction * 3,
    side_1=0,
    side_2=0,
)

# Ovals can be created by setting side_0 or side_1 different from 0.
oval_sketch_1 = cb.SplineDisk(
    center_point=center_point + direction * 3.4,
    corner_1_point=np.asarray([0, 1.2, 0]) + direction * 3.4,
    corner_2_point=np.asarray([0, 0, 2.2]) + direction * 3.4,
    side_1=0.3,
    side_2=0.3,
)


oval_sketch_2 = cb.SplineDisk(
    center_point=center_point + direction * 4,
    corner_1_point=np.asarray([0, 1.5, 0]) + direction * 4,
    corner_2_point=np.asarray([0, 0, 2.5]) + direction * 4,
    side_1=0.8,
    side_2=1.8,
    n_straight_spline_points=100,
)
# It is also possible to create a hollow ring.
# The ring is defined as the disk,
# where a ring with same center, corner_1... as a disk will fit on the outside of the disk.
oval_ring_1 = cb.SplineRing(
    center_point=oval_sketch_2.center,
    corner_1_point=oval_sketch_2.radius_1_point,
    corner_2_point=oval_sketch_2.radius_2_point,
    side_1=oval_sketch_2.side_1,
    side_2=oval_sketch_2.side_2,
    width_1=0.2,
    width_2=0.2,
    n_outer_spline_points=100,
)

# Note it is possible to access the center and corners of a defined sketch. These are stable on transformation.
oval_ring_2 = cb.SplineRing(
    center_point=oval_sketch_2.center + direction,
    corner_1_point=oval_sketch_2.radius_1_point + direction,
    corner_2_point=oval_sketch_2.radius_2_point + direction,
    side_1=0,
    side_2=0,
    width_1=0.2,
    width_2=0.2,
    n_outer_spline_points=100,
)

# A lofted shape with a list of sketches as midSketch,
# creates splines going through the midSketches in the Lofted direction.
shape_1 = cb.LoftedShape(circular_sketch_1, elliptical_sketch_3, [elliptical_sketch_1, elliptical_sketch_2])

# A lofted shape with a only one of sketches as midSketch,
# creates arches going through the midSketches in the Lofted direction.
shape_2 = cb.LoftedShape(elliptical_sketch_3, oval_sketch_2, [oval_sketch_1])

# A lofted shape without midSketch,
# creates straight lines in the Lofted direction.
shape_3 = cb.LoftedShape(oval_ring_1, oval_ring_2)

shape_1.chop(0, start_size=0.05)
shape_1.chop(1, start_size=0.05)
shape_1.chop(2, start_size=0.05)

shape_2.chop(2, start_size=0.05)

shape_3.chop(0, start_size=0.05)
shape_3.chop(2, start_size=0.05)


mesh = cb.Mesh()
mesh.add(shape_1)
mesh.add(shape_2)
mesh.add(shape_3)
mesh.write(os.path.join("..", "..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
