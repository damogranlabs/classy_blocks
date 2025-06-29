#!/usr/bin/env python
import os
import time

import numpy as np
import scipy.optimize
from involute_gear import InvoluteGear
from tooth import ToothSketch

import classy_blocks as cb
from classy_blocks.util import functions as f

time_start = time.time()

mesh = cb.Mesh()

# parameters
RADIUS_EXPANSION = 1.1

# create an interpolated curve that represents a gear tooth
fillet = 0.1  # Must be more than zero!
gear = InvoluteGear(fillet=fillet, arc_step_size=0.1, max_steps=1000, teeth=15)
tng_points = gear.generate_tooth_and_gap()

z_coords = np.zeros(len(tng_points[0]))
tng_points = np.stack((tng_points[0], tng_points[1], z_coords)).T
tng_points = np.flip(tng_points, axis=0)

# add start and end points exactly on the 2pi/teeth
start_point = f.to_polar(tng_points[0], axis="z")
start_point[1] = np.pi / gear.teeth
start_point = f.to_cartesian(start_point)

end_point = f.to_polar(tng_points[-1], axis="z")
end_point[1] = -np.pi / gear.teeth
end_point = f.to_cartesian(end_point)

tng_points = np.concatenate(([start_point], tng_points, [end_point]))

tooth_curve = cb.LinearInterpolatedCurve(tng_points)
gear_params = np.linspace(0, 1, num=8)


# fix points 1 and 3:
# 1 is on radius gear.root_radius + fillet/2
def frad(t, radius):
    return f.norm(tooth_curve.get_point(t)) - radius


gear_params[1] = scipy.optimize.brentq(lambda t: frad(t, gear.root_radius + fillet / 2), 0, 0.25)

# 3 is on radius gear.outer_radius - fillet/2
gear_params[3] = scipy.optimize.brentq(lambda t: frad(t, gear.outer_radius - fillet / 2), 0.25, 0.5)

gear_params[6] = 1 - gear_params[1]
gear_params[4] = 1 - gear_params[3]

gear_params[2] = (gear_params[1] + gear_params[3]) / 2
gear_params[5] = (gear_params[4] + gear_params[6]) / 2

gear_points = np.array([tooth_curve.get_point(t) for t in gear_params])

outer_radius = f.norm(gear_points[3] * RADIUS_EXPANSION)
p11_polar = f.to_polar(gear_points[-1], axis="z")
p14_polar = f.to_polar(gear_points[0], axis="z")
angles = np.linspace(p11_polar[1], p14_polar[1], num=4)
tangential_points = np.array([f.to_cartesian([outer_radius, angle, 0]) for angle in angles])

radial_points_1 = np.linspace(gear_points[-1], tangential_points[0], axis=0, num=5)[1:-1]
radial_points_2 = np.linspace(tangential_points[-1], gear_points[0], axis=0, num=5)[1:-1]

outer_points = np.concatenate((gear_points, radial_points_1, tangential_points, radial_points_2))
inner_points = np.zeros((6, 3))

positions = np.concatenate((outer_points, inner_points))

# At this point, a smoother would reliably
# produce almost the best blocking if this was a convex sketch.
# Alas, this is a severely concave case so smoothing will produce
# degenerate quads which even optimizers won't be able to fix.
# It's best to manually position points, then optimize the sketch.


def mirror(target, source):
    # once a position is obtained, the mirrored counterpart is also determined
    positions[target] = [positions[source][0], -positions[source][1], 0]


# fix points 18 and 23 because optimization doesn't 'see' curved edges
# and produces high non-orthogonality
dir_0 = f.unit_vector(gear_points[0] - gear_points[1])
dir_2 = f.unit_vector(gear_points[2] - gear_points[1])
dir_18 = f.unit_vector(dir_0 + dir_2)

positions[18] = gear_points[1] + dir_18 * f.norm(gear_points[0] - gear_points[1]) / 2
mirror(23, 18)

positions[17] = positions[0] + f.unit_vector(positions[17] - positions[0]) * f.norm(positions[18] - positions[1])
mirror(8, 17)


# other points are somewhere between...
def midpoint(target, left, right):
    positions[target] = (positions[left] + positions[right]) / 2


midpoint(19, 2, 16)
midpoint(20, 3, 13)

# and their mirrored counterparts
mirror(22, 19)
mirror(21, 20)


sketch = ToothSketch(positions, tooth_curve)

# Optimize the sketch:
optimizer = cb.SketchOptimizer(sketch)

# point 2 is on gear curve
optimizer.add_clamp(cb.CurveClamp(positions[2], tooth_curve))

# point 13 is movable radially
optimizer.add_clamp(cb.RadialClamp(positions[13], [0, 0, 0], [0, 0, 1]))

# 15-17 move along a line
for i in (15, 16, 17):
    optimizer.add_clamp(cb.LineClamp(positions[i], gear_points[0], tangential_points[-1]))

# freely movable points (on sketch plane)
# TODO: easier clamp definition for sketch optimization
for i in (19, 20):
    optimizer.add_clamp(cb.PlaneClamp(sketch.positions[i], sketch.positions[i], sketch.normal))

# Links!
symmetry_pairs = [
    (2, 5),
    (19, 22),
    (20, 21),
    (17, 8),
    (16, 9),
    (15, 10),
]

for pair in symmetry_pairs:
    optimizer.add_link(cb.SymmetryLink(positions[pair[0]], positions[pair[1]], f.vector(0, 1, 0), f.vector(0, 0, 0)))

optimizer.optimize()

stack = cb.TransformedStack(
    sketch,
    [cb.Translation([0, 0, 4]), cb.Rotation(sketch.normal, 10 * np.pi / 180, [0, 0, 0])],
    2,
    [cb.Translation([0, 0, 2]), cb.Rotation(sketch.normal, 5 * np.pi / 180, [0, 0, 0])],
)

# TODO: this be mighty clumsy; unclumsify
bulk_size = 0.1
wall_size = 0.01

stack.shapes[0].chop(0, start_size=bulk_size)
stack.shapes[0].chop(1, start_size=wall_size, end_size=bulk_size / 2)
stack.shapes[0].operations[10].chop(1, start_size=bulk_size)
stack.chop(count=8)

# patches
for shape in stack.shapes:
    for i in range(7):
        shape.operations[i].set_patch("front", "gear")

    for i in (9, 10, 11):
        shape.operations[i].set_patch("front", "outer")

for operation in stack.shapes[0].operations:
    operation.set_patch("bottom", "bottom")

for operation in stack.shapes[-1].operations:
    operation.set_patch("top", "top")

mesh.add(stack)

for i, angle in enumerate(np.linspace(0, 2 * np.pi, num=gear.teeth, endpoint=False)[1:]):
    print(f"Adding tooth {i + 2}")
    mesh.add(stack.copy().rotate(angle, [0, 0, 1], [0, 0, 0]))

print("Writing mesh...")
mesh.write(os.path.join("..", "..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")

time_end = time.time()

print(f"Elapsed time: {time_end - time_start}s")
