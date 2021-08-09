from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.shapes import ExtrudedRing, Box

cylinder_diameter = 20e-3 # [m]
ring_thickness = 5e-3 # [m]

# increase domain height for "proper" simulation
domain_height = 0.05 # [m]
upstream_length = 0.03 # [m]
downstream_length = 0.05 # [m]

# size to roughly match cells outside ring
cell_size = 0.3*ring_thickness

# it's a 2-dimensional case
z = 0.01

mesh = Mesh()

# a layer of cells on the cylinder
d = 2**0.5/2
ring_point = d*cylinder_diameter/2
outer_point = d*(cylinder_diameter/2 + ring_thickness)

wall_ring = ExtrudedRing(
    [0, 0, 0],
    [0, 0, z],
    [ring_point, ring_point, 0],
    cylinder_diameter/2 + ring_thickness
)

wall_ring.set_axial_cell_count(1)
wall_ring.set_radial_cell_count(10)
wall_ring.count_to_size_tangential(ring_thickness/3)

wall_ring.grade_to_size_radial(0.001)

wall_ring.set_inner_patch('cylinder')

mesh.add(wall_ring)

# boxes that fill up the whole domain
def make_box(p1, p2, size_axes, patches):
    box = Box(
        [p1[0], p1[1], 0],
        [p2[0], p2[1], z])

    for axis in size_axes:
        box.count_to_size(axis, cell_size)
    
    for side, name in patches.items():
        box.set_patch(side, name)

    mesh.add(box)

# top 3 boxes
make_box(
    [-upstream_length, outer_point],
    [-outer_point, domain_height/2],
    [0, 1],
    {'back': 'upper_wall', 'left': 'inlet'})
make_box(
    [-outer_point, outer_point],
    [outer_point, domain_height/2],
    [],
    {'back': 'upper_wall'})
make_box(
    [outer_point, outer_point],
    [downstream_length, domain_height/2],
    [0, 1],
    {'back': 'upper_wall', 'right': 'outlet'})

# left and right of the cylinder
make_box(
    [-upstream_length, -outer_point],
    [-outer_point, outer_point],
    [],
    {'left': 'inlet'})
make_box(
    [outer_point, -outer_point],
    [downstream_length, outer_point],
    [],
    {'right': 'outlet'})

# bottom 3 boxes
make_box(
    [-upstream_length, -domain_height/2],
    [-outer_point, -outer_point],
    [0, 1],
    {'front': 'lower_wall', 'left': 'inlet'})
make_box(
    [-outer_point, -domain_height/2],
    [outer_point, -outer_point],
    [],
    {'front': 'lower_wall'})
make_box(
    [outer_point, -domain_height/2],
    [downstream_length, -outer_point],
    [0, 1],
    {'front': 'lower_wall', 'right': 'outlet'})

mesh.write('case/system/blockMeshDict')