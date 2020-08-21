import os
import numpy as np

from classes.primitives import Edge
from classes.mesh import Mesh

from operations.base import Face
from shapes.shapes import Cylinder, Elbow

from util.methematics import functions as g

def create():
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
    l_connection = 100e-3
    l_outlet = 500e-3

    # number of cells
    n_cells_radial = 10
    n_cells_tangential = 12

    # cell sizes to adjust cell count
    cell_size = (2*r_pipe)/(3*n_cells_radial)
    # ratio of cell length (flow diretion) vs. cell height;
    # used to lower cell count where nothing interesting happens
    cell_ratio = 3

    # shapes
    inlet = Cylinder([0, 0, 0], [l_inlet, 0, 0], [0, r_pipe, 0])
    inlet.set_bottom_patch('inlet')

    # set cell counts on all directions - for first block/operation/shape only!
    inlet.set_axial_cell_count(l_inlet/cell_size/cell_ratio)
    inlet.set_radial_cell_count(n_cells_radial)
    inlet.set_tangential_cell_count(n_cells_tangential)

    # grade to desired cell size
    inlet.set_axial_cell_size(-cell_size) # negative size means we're setting at the other end

    elbow_1 = Elbow(
        inlet.axis_point_2, 
        inlet.radius_point_2,
        inlet.axis, 
        np.pi/2, # sweep angle
        [l_inlet, r_elbow, 0],  # arc center
        [0, 0, 1], # rotation axis
        r_pipe # radius 2
    )
    elbow_cell_count = (np.pi/2)*(r_elbow+r_pipe)/cell_size
    elbow_1.set_axial_cell_count(elbow_cell_count)
    

    connection = Cylinder(
        elbow_1.circle_2.center_point,
        elbow_1.circle_2.center_point + np.array([0, l_connection, 0]),
        elbow_1.circle_2.radius_point
    )
    connection.set_axial_cell_count(l_connection/cell_size)
    
    elbow_2 = Elbow(
        connection.axis_point_2,
        connection.radius_point_2,
        connection.axis,
        np.pi/2,
        connection.axis_point_2 + np.array([0, 0, r_elbow]),
        [1, 0, 0],
        r_pipe
    )
    elbow_2.set_axial_cell_count(elbow_cell_count)

    outlet = Cylinder(
        elbow_2.circle_2.center_point,
        elbow_2.circle_2.center_point + np.array([0, 0, l_outlet]),
        elbow_2.circle_2.radius_point
    )
    outlet.set_top_patch('outlet')
    outlet.set_axial_cell_count(l_outlet/cell_size/cell_ratio)
    outlet.set_axial_cell_size(cell_size)
    
    ### adjust other, common stuff and add shapes to mesh
    mesh = Mesh()

    for shape in [inlet, elbow_1, connection, elbow_2, outlet]:
        shape.set_outer_patch('wall')
        mesh.add_shape(shape)

    mesh.write('util/blockMeshDict.template', 'examples/meshCase/system/blockMeshDict')

    # run blockMesh
    os.system("blockMesh -case examples/meshCase")