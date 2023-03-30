import os

from classy_blocks import Cylinder, ExtrudedRing, Mesh

# A mesh for calculation of friction losses of a rotating rotor in a fluid.
# See flywheel.svg for explanation of dimensions and block numbers;
r_wheel = 8
t_rim = 1
w_rim = 3
w_wheel = 1

h_gap = 1
w_gap = 1

r_shaft = 1

# cell sizing
cell_size = 0.2
bl_thickness = 0.05
c2c_expansion = 1.25

# patches
fixed_patch = 'fixed_wall'
rotating_patch = 'rotating_wall'

mesh = Mesh()

# put all shapes into this list for easier 'handling';
# indexes refer to block numbers in sketch
shapes = []

# block 0
shapes.append(Cylinder(
    [0, 0, 0], # axis point 1
    [w_gap, 0, 0], # axis point 2
    [0, r_wheel - t_rim, 0] # radius point 1
))
shapes.append(Cylinder.chain(shapes[0], (w_rim - w_wheel)/2))
shapes.append(ExtrudedRing.expand(shapes[0], t_rim))
shapes.append(ExtrudedRing.expand(shapes[2], h_gap))
shapes.append(ExtrudedRing.chain(shapes[3], w_rim))
shapes.append(ExtrudedRing.chain(shapes[4], w_gap))
shapes.append(ExtrudedRing.contract(shapes[5], r_wheel - t_rim))
shapes.append(ExtrudedRing.contract(shapes[6], r_shaft))
shapes.append(ExtrudedRing.chain(shapes[7], (w_rim - w_wheel)/2, start_face=True))

# Chopping:
# Cells on fixed wall are cell_size;
# rotating wall (wheel) is resolved, thus bl_thickness.
# Depending on geometry, two different scenarios are possible:
#  a) use end_size and c2c_expansion or
#  b) start_size and end_size
# In a), too large cells might be obtained far away from the wall, or
# in b), to high cell-to-cell expansion might be made.

def double_chop(function):        
    function(length_ratio=0.5, start_size=bl_thickness, c2c_expansion=c2c_expansion)
    function(length_ratio=0.5, end_size=bl_thickness, c2c_expansion=1/c2c_expansion)

# Axial
for i in (1, 2, 4, 6, 8):
    double_chop(shapes[i].chop_axial)

# Radial
for i in (2, 4, 6, 7):
    double_chop(shapes[i].chop_radial)

shapes[0].chop_radial(end_size=bl_thickness, c2c_expansion=1/c2c_expansion)

# adjust this cell size to obtain roughly the same cell sizes
# in core and shell of the cylinder
shapes[2].chop_tangential(start_size=4*cell_size)

# Patch names:
for i in (0, 2, 3):
    shapes[i].set_start_patch(fixed_patch)

for i in (3, 4, 5):
    shapes[i].set_outer_patch(fixed_patch)

for i in (5, 6, 7):
    shapes[i].set_end_patch(fixed_patch)

for shape in shapes:
    mesh.add(shape)

mesh.set_default_patch(rotating_patch, 'wall')
mesh.modify_patch(fixed_patch, 'wall')

mesh.write(os.path.join('..', 'case', 'system', 'blockMeshDict'), debug_path='debug.vtk')
