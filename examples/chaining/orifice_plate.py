from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.shapes import Cylinder, Frustum, ExtrudedRing

def get_mesh():
    # see orifice_plate.svg for sketch

    D = 0.1 # [m]
    d_0 = 0.025
    t = 0.005 # thickness of orifice plate

    entry_length = 5 # *D
    exit_length = 8 # *D
    
    # chopping
    cell_size = t # use much less in real life
    # to save on simulation time
    # use bigger cells in lenghty entry/exit sections
    cell_dilution = 2

    mesh = Mesh()

    r_0 = d_0/2
    r_1 = r_0 + t/4
    r_2 = r_0 + t/2

    shapes = [None]*7
    shapes[0] = Cylinder(
        [0, 0, 0],
        [D*entry_length, 0, 0],
        [0, r_1, 0])

    shapes[1] = Frustum.chain(shapes[0], t/4, r_0)
    shapes[2] = Cylinder.chain(shapes[1], t/4)
    shapes[3] = Frustum.chain(shapes[2], t/2, r_2)
    shapes[4] = Cylinder.chain(shapes[3], D*exit_length)
    shapes[5] = ExtrudedRing.expand(shapes[0], D/2 - r_1)
    shapes[6] = ExtrudedRing.expand(shapes[4], D/2 - r_2)

    # chop to a sensible number of cells;
    # dilute long inlet/outlet sections
    shapes[0].chop_radial(start_size=cell_size)
    shapes[0].chop_tangential(start_size=cell_size)
    shapes[0].chop_axial(length_ratio=0.9, start_size=cell_size*cell_dilution, end_size=cell_size)
    shapes[0].chop_axial(length_ratio=0.1, start_size=cell_size, end_size=t/8)

    shapes[6].chop_axial(length_ratio=0.1, start_size=t/8, end_size=cell_size)
    shapes[6].chop_axial(length_ratio=0.9, start_size=cell_size, end_size=cell_size*cell_dilution)


    shapes[5].chop_radial(start_size=cell_size/2, end_size=2*cell_size)
    shapes[6].chop_radial(start_size=cell_size/2, end_size=2*cell_size)

    for i in (1, 2, 3):
        shapes[i].chop_axial(start_size=t/8)

    for s in shapes:
        if s is not None:
            mesh.add(s)

    # patches
    for i in (0, 5):
        shapes[i].set_bottom_patch('inlet')
    for i in (4, 6):
        shapes[i].set_top_patch('outlet')
    
    mesh.set_default_patch('walls', 'wall')
    
    return mesh
