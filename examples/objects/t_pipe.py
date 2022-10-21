from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.objects import Semicylinder, SharpSemicylinder

import numpy as np

# a T-joint:
#            
# left <---- O ----> right (x-coordinate)
#            ^
#            |
#            |
#            | inlet
#
# O = origin
# y-coordinate points upwards, z-coordinate out from the screen


inlet_point = [0, -1, 0]
center_point = [0, 0, 0]
arm_length = 1
pipe_diameter = 0.5

def get_mesh():
    mesh = Mesh()

    inlet_start_point = inlet_point
    inlet_left = SharpSemicylinder(inlet_start_point, center_point, [0, -1, pipe_diameter/2])

    inlet_right = SharpSemicylinder(inlet_start_point, center_point, [0, -1, -pipe_diameter/2])

    left_center = [-arm_length, 0, 0]
    left_lower = SharpSemicylinder(left_center, center_point, [-arm_length, 0, -pipe_diameter/2])

    left_upper = Semicylinder(left_center, center_point, [-arm_length, 0, pipe_diameter/2])

    right_center = [arm_length, 0, 0]
    right_lower = SharpSemicylinder(right_center, center_point, [arm_length, 0, pipe_diameter/2])

    right_upper = Semicylinder(right_center, center_point, [arm_length, 0, -pipe_diameter/2])

    #inlet_left.set_bottom_patch('inlet')
    #inlet_left.set_top_patch('outlet')
    #inlet_left.set_outer_patch('walls')

    #bl_thickness=0.05
    #core_size = 0.2

    #inlet_left.chop_axial(count=30)
    #inlet_left.chop_radial(start_size=core_size, end_size=bl_thickness)
    #inlet_left.chop_tangential(start_size=core_size)


    inlet_left.chop_axial(count=10)
    inlet_left.chop_radial(count=5)
    inlet_left.chop_tangential(count=5)
    mesh.add(inlet_left)

    inlet_right.chop_tangential(count=5)
    mesh.add(inlet_right)

    left_lower.chop_axial(count=10)
    mesh.add(left_lower)

    left_upper.chop_tangential(count=5)
    mesh.add(left_upper)

    right_lower.chop_axial(count=10)
    mesh.add(right_lower)

    mesh.add(right_upper)

    return mesh
