import os
import numpy as np

from classy_blocks import Cylinder, Frustum, Mesh
from classy_blocks.util import functions as f

def calculate_fillet(r_pipe, r_fillet, a_cone):
    # see venturi_tube.svg for explanation
    a_cone = f.deg2rad(a_cone)
    y_center = r_pipe - r_fillet
    l_f = abs(r_fillet*np.sin(a_cone))

    r_2 = y_center + r_fillet*np.cos(a_cone)
    r_mid = y_center + r_fillet*np.cos(a_cone/2)

    return l_f, r_2, r_mid

def calculate_cone(r_start, r_end, a_cone):
    return abs(r_end - r_start)/np.tan(f.deg2rad(a_cone))

# see venturi_tube.svg for sketch
# https://www.researchgate.net/figure/The-Critical-Dimensions-of-the-Classical-Venturi-Tube-Source-p-24-Principles-and_fig5_311949745

D = 0.1 # [m]
d = 0.03 # exaggerated to illustrate the awesomeness
entry_length = 5 # *D
entry_angle = 20 # degrees

exit_length = 8 # *D
exit_angle = 10 # degrees

fillet_radius = 1.5*D # also exaggerated to display even more awesomeness

# chopping
cell_size = 0.08*D
# to save on simulation time
# use bigger cells in lenghty entry/exit sections
cell_dilution = 5 

mesh = Mesh()

shapes = []
# entry tube
shapes.append(Cylinder(
    [0, 0, 0],
    [D*entry_length, 0, 0],
    [0, D/2, 0]))

# Contraction: two fillets and a cone in between
# fillet from entry cylinder to entry cone
l_fillet_1, r_fillet_1, r_fillet_1_mid = calculate_fillet(D/2, fillet_radius, entry_angle/2)
# fillet from entry cone to middle cylinder
l_fillet_2, r_fillet_2, r_fillet_2_mid = calculate_fillet(d/2, -fillet_radius, entry_angle/2)
# length of cone connecting the fillets
l_cone = calculate_cone(r_fillet_1, r_fillet_2, entry_angle/2) - l_fillet_1 - l_fillet_2
# print(l_fillet_1, l_cone, l_fillet_2)
# print(r_fillet_1, r_fillet_2)

shapes.append(Frustum.chain(shapes[-1], l_fillet_1, r_fillet_1, r_fillet_1_mid))
shapes.append(Frustum.chain(shapes[-1], l_cone, r_fillet_2))
shapes.append(Frustum.chain(shapes[-1], l_fillet_2, d/2, r_fillet_2_mid))

# the narrowest part
shapes.append(Cylinder.chain(shapes[-1], d))

# expansion:
# same as contraction but at different angle
l_fillet_3, r_fillet_3, r_fillet_3_mid = calculate_fillet(d/2, -fillet_radius, exit_angle/2)
l_fillet_4, r_fillet_4, r_fillet_4_mid = calculate_fillet(D/2, fillet_radius, exit_angle/2)
l_cone = calculate_cone(r_fillet_3, r_fillet_4, exit_angle/2) - l_fillet_3 - l_fillet_4
# print(l_fillet_3, l_cone, l_fillet_4)
# print(r_fillet_3, r_fillet_4)

shapes.append(Frustum.chain(shapes[-1], l_fillet_3, r_fillet_3, r_fillet_3_mid))
shapes.append(Frustum.chain(shapes[-1], l_cone, r_fillet_4))
shapes.append(Frustum.chain(shapes[-1], l_fillet_4, D/2, r_fillet_4_mid))
shapes.append(Cylinder.chain(shapes[-1], D*exit_length))

# all cells sizes in longitudinal direction are fixed by the first block
shapes[0].chop_radial(start_size=cell_size)
shapes[0].chop_tangential(start_size=cell_size)        

# use smaller cells in smaller diameters
for s in shapes[1:-1]:
    s.chop_axial(start_size=cell_size*s.sketch_1.radius*2/D,
        end_size=cell_size*s.sketch_2.radius*2/D)

# dilute cells in first and last block
shapes[0].chop_axial(start_size=cell_size*cell_dilution, end_size=cell_size)
shapes[-1].chop_axial(end_size=cell_size*cell_dilution, start_size=cell_size)

# patches
shapes[0].set_start_patch('inlet')
shapes[-1].set_end_patch('outlet')
mesh.set_default_patch('walls', 'wall')

for s in shapes:
    mesh.add(s)

mesh.write(os.path.join('..', 'case', 'system', 'blockMeshDict'), debug_path='debug.vtk')
