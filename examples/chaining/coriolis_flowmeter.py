import numpy as np

from classy_blocks import Cylinder, Elbow, ExtrudedRing, ElbowWall, Mesh
from classy_blocks.util import functions as f

def get_mesh():
    # Coriolis flow meters have this specific pipe shape that's
    # approximately modeled here;
    # this mesh has two zones, 'fluid' and 'solid' and is
    # ready-ish-to-use on a FSI (fluid-structure interaction) simulation.

    mesh = Mesh()

    # geometry data
    d_pipe = 20
    t_wall = 1.5

    entry_length = 5 # times d_pipe
    elbow_angle = 120 # degrees
    elbow_radius = d_pipe

    d_c = 10 # the pipe constracts in 'elbow'
    l_c = 120 # length of straight 'measurement' section
    l_y = 40 # length of angled pipe

    exit_length = 8 # times d_pipe

    # cell sizing
    cell_size = d_pipe/10
    cell_expansion = 5 # larger cells in inlet/outlet
    bl_thickness = 0.15

    # pre-calculated stuff
    r_pipe = d_pipe/2
    r_c = d_c/2
    r_mid = (r_pipe + r_c)/2
    elbow_angle = f.deg2rad(elbow_angle)

    # shapes
    fluid_shapes = []

    fluid_shapes.append(Cylinder([0, 0, -d_pipe*entry_length], [0, 0, 0], [0, r_pipe, -d_pipe*entry_length]))

    # first, contracting elbow
    # Note: cross-section of elbows isn't circular everywhere;
    # that's because there are 'Loft'ed blocks on the outside and
    # only their edges can be defined. Number of outside blocks can't be
    # increased because it's conditioned by inside blocks - but you can
    # assemble an elbow from smaller elbows with smaller swipe angles
    # to gain precision if that's important.
    # 90 or 120 degree is too much so we'll make the turns in 2 steps
    center_1 = f.rotate(f.vector(elbow_radius, 0, 0), elbow_angle, axis='z')
    axis_1 = f.rotate(center_1, np.pi/2, axis='z')
    fluid_shapes.append(Elbow.chain(fluid_shapes[-1], np.pi/4, center_1, axis_1, r_mid))
    fluid_shapes.append(Elbow.chain(fluid_shapes[-1], np.pi/4, center_1, axis_1, r_c))

    # the 'inclined' connection pipe
    fluid_shapes.append(Cylinder.chain(fluid_shapes[-1], l_y))

    # get the new radius from cylinder's end face
    s = fluid_shapes[-1].sketch_2
    center_2 = s.center_point + f.unit_vector(s.radius_vector)*elbow_radius
    fluid_shapes.append(Elbow.chain(fluid_shapes[-1], -elbow_angle/2, center_2, [0, 0, 1], r_c))
    fluid_shapes.append(Elbow.chain(fluid_shapes[-1], -elbow_angle/2, center_2, [0, 0, 1], r_c))

    # the 'measuring pipe'
    fluid_shapes.append(Cylinder.chain(fluid_shapes[-1], l_c))

    # 2 elbows for the other turn
    s = fluid_shapes[-1].sketch_2
    center_3 = s.center_point + f.unit_vector(s.radius_vector)*elbow_radius
    fluid_shapes.append(Elbow.chain(fluid_shapes[-1], -elbow_angle/2, center_3, [0, 0, 1], r_c))
    fluid_shapes.append(Elbow.chain(fluid_shapes[-1], -elbow_angle/2, center_3, [0, 0, 1], r_c))

    # connection 2
    fluid_shapes.append(Cylinder.chain(fluid_shapes[-1], l_y))

    # expanding elbows back to d_pipe
    s = fluid_shapes[-1].sketch_2
    center_4 = s.center_point + f.vector(0, 0, -elbow_radius)
    axis_4 = f.rotate(f.vector(0, 1, 0), -elbow_angle, axis='z')
    fluid_shapes.append(Elbow.chain(fluid_shapes[-1], np.pi/4, center_4, axis_4, r_mid))
    fluid_shapes.append(Elbow.chain(fluid_shapes[-1], np.pi/4, center_4, axis_4, r_pipe))

    # outlet pipe
    fluid_shapes.append(Cylinder.chain(fluid_shapes[-1], d_pipe*exit_length))
    
    # only the first
    fluid_shapes[0].chop_tangential(start_size=cell_size)
    fluid_shapes[0].chop_radial(start_size=cell_size, end_size=bl_thickness)

    for fs in fluid_shapes[1:-1]:
        fs.chop_axial(start_size=cell_size)

    # uess less cells in straight pipes
    fluid_shapes[0].chop_axial(start_size=cell_size*cell_expansion, end_size=cell_size)
    fluid_shapes[-1].chop_axial(start_size=cell_size, end_size=cell_size*cell_expansion)

    # add pipe walls
    solid_shapes = []

    for fs in fluid_shapes:
        fs.set_cell_zone('fluid')

        if fs.__class__ == Cylinder:
            solid_class = ExtrudedRing
        elif fs.__class__ == Elbow:
            solid_class = ElbowWall
        
        solid = solid_class.expand(fs, t_wall)
        solid.set_cell_zone('solid')
        solid_shapes.append(solid)
    
    solid_shapes[0].chop_radial(count=4)

    for s in fluid_shapes + solid_shapes:
        mesh.add(s)

    # patches
    fluid_shapes[0].set_bottom_patch('inlet')
    fluid_shapes[-1].set_top_patch('outlet')

    solid_shapes[0].set_bottom_patch('pipe_start')
    solid_shapes[-1].set_top_patch('pipe_end')

    mesh.set_default_patch('pipe_outer', 'wall')

    return mesh
