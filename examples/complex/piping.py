import numpy as np

from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.shapes import Cylinder, Elbow

def get_mesh():
    # a pipe with two 90-degree elbows:
    # to demonstrate vortex creation in consecutive elbows in different planes;

    # Notice: cross-section of elbows isn't circular everywhere;
    # that's because there are 4 'Loft'ed blocks on the outside and
    # edges are defined on 4 lines only. Number of outside blocks can't be
    # increased because there's an internal 4-sided block - but you can
    # assemble an elbow from smaller elbows with smaller swipe angles
    # to gain precision if that's important.

    # geometry data (all dimensions in meters):
    r_pipe = 50e-3
    r_elbow = 60e-3
    l_inlet = 300e-3
    l_connection = 200e-3
    l_outlet = 500e-3

    # cell sizing
    min_size = r_pipe/8
    max_size = 5*min_size # use bigger cells around not-so-important parts

    # shapes
    inlet = Cylinder([0, 0, 0], [l_inlet, 0, 0], [0, r_pipe, 0])
    inlet.set_bottom_patch('inlet')

    # set cell counts on all directions - for first block/operation/shape only!
    inlet.chop_tangential(start_size=min_size) # this is propagated along all other shapes
    inlet.chop_radial(start_size=min_size/2) # this too
    inlet.chop_axial(start_size=max_size, end_size=min_size)

    elbow_1 = Elbow(
        inlet.axis_point_2, 
        inlet.radius_point_2,
        inlet.axis, 
        np.pi/2, # sweep angle
        [l_inlet, r_elbow, 0],  # arc center
        [0, 0, 1], # rotation axis
        r_pipe # radius 2
    )
    elbow_1.chop_axial(start_size=min_size)

    connection = Cylinder(
        elbow_1.circle_2.center_point,
        elbow_1.circle_2.center_point + np.array([0, l_connection, 0]),
        elbow_1.circle_2.radius_point
    )
    connection.chop_axial(length_ratio=0.5, start_size=min_size, c2c_expansion=1.2)
    connection.chop_axial(length_ratio=0.5, end_size=min_size, c2c_expansion=1/1.2)

    elbow_2 = Elbow(
        connection.axis_point_2,
        connection.radius_point_2,
        connection.axis,
        np.pi/2,
        connection.axis_point_2 + np.array([0, 0, r_elbow]),
        [1, 0, 0],
        r_pipe
    )
    elbow_2.chop_axial(start_size=min_size)

    outlet = Cylinder(
        elbow_2.circle_2.center_point,
        elbow_2.circle_2.center_point + np.array([0, 0, l_outlet]),
        elbow_2.circle_2.radius_point
    )
    outlet.set_top_patch('outlet')
    outlet.chop_axial(start_size=min_size, end_size=max_size)

    ### adjust other, common stuff and add shapes to mesh
    mesh = Mesh()

    for shape in [inlet, elbow_1, connection, elbow_2, outlet]:
        shape.set_outer_patch('wall')
        mesh.add(shape)

    return mesh
