import os

import numpy as np

from classes.mesh import Mesh
from classes.block import Block
from util import geometry as g

def create():
    # geometry:
    # ----d_w-------------------------   < static wall
    # |           >annulus>          |
    # |>inlet   |----d_r----| outlet>|
    # |         | < rotor > |        |
    # |---d_s---|           |--------|
    #
    # - ----- - ----- - ----- - ----- - ----- CofR - x-axis

    d_s = 8.0e-3 # [m], shaft diameter
    d_w = 28.0e-3 # [m], outside (wall) diameter
    d_r = 24.0e-3 # [m], inside (rotor) diameter

    l_inlet = 10.0e-3 # [m], inlet length
    l_rotor = 15.0e-3 # [m]
    l_outlet = 20.0e-3 # [m], outlet length

    n_segments = 4 # number of segments (blocks) in a ring

    # cell sizes
    n_cells_radial = 100
    n_cells_annulus = 10 # number of cells in annulus, radially
    cell_length = 5 # length of cell with respect to its height

    # number of cells in inlet and outlet blocks will be 'diluted' and graded to match annulus size
    axial_dilution = 2

    ### create a classy mesh
    mesh = Mesh()

    # calculate cell size in annulus; other cells depend on this
    cell_size_radial = (d_w - d_r)/2/n_cells_annulus
    cell_size_axial = cell_length*cell_size_radial

    def ring(x_1, x_2, r_1, r_2, patch_inner=None, patch_outer=None, patch_left=None, patch_right=None):
        angles = np.linspace(0, 2*np.pi, num=n_segments+1)

        # these points will be revolved around x-axis to create segments
        points = [
            g.vector(x_1, r_1, 0),
            g.vector(x_2, r_1, 0),
            g.vector(x_2, r_2, 0),
            g.vector(x_1, r_2, 0)
        ]

        def ring_segment(start_angle, end_angle):
            points_front = [g.rotate(p, start_angle) for p in points]
            points_back = [g.rotate(p, end_angle) for p in points]

            vertices = mesh.add_vertices(points_back + points_front)
            block = mesh.add_block(vertices, [
                int((x_2 - x_1)/cell_size_axial), # 
                n_cells_annulus, # 
                int(n_cells_radial/n_segments) #
                ])

            edge_points = [g.rotate(p, (start_angle - end_angle)/2) for p in points_front]

            for i in range(4):
                # add edges existing vertices
                mesh.add_edge(vertices[i], vertices[i+4], edge_points[i])

            # patches:
            if patch_inner:
                block.set_patch(['top'], patch_inner)

            if patch_outer:
                block.set_patch(['bottom'], patch_outer)
            
            if patch_left: # a.k.a. 'inlet' patch
                block.set_patch(['front'], patch_left)

            if patch_right: # a.k.a. 'outlet' patch
                block.set_patch(['back'], patch_right)

            return block

        ring_blocks = []

        for i in range(n_segments):
            ring_blocks.append(ring_segment(angles[i], angles[i+1]))

        return ring_blocks


    # inlet
    x_start = 0
    x_end = l_inlet
    inlet_inner = ring(x_start, x_end, d_s/2, d_r/2, patch_left='inlet', patch_inner='rotor', patch_right='rotor')
    inlet_outer = ring(x_start, x_end, d_r/2, d_w/2, patch_left='inlet', patch_outer='stator')

    for i in range(n_segments):
        inlet_inner[i].set_cell_size(1, cell_size_radial)

        inlet_inner[i].n_cells[0] = int(inlet_inner[i].n_cells[0]/axial_dilution)
        inlet_inner[i].set_cell_size(0, cell_size_axial)

        inlet_outer[i].n_cells[0] = int(inlet_outer[i].n_cells[0]/axial_dilution)
        inlet_outer[i].set_cell_size(0, cell_size_axial)

    x_start = x_end
    x_end = x_end + l_rotor
    annulus = ring(x_start, x_end, d_r/2, d_w/2, patch_inner='rotor', patch_outer='stator')

    x_start = x_end
    x_end = x_end + l_outlet
    outlet_outer = ring(x_start, x_end, d_r/2, d_w/2, patch_outer='stator', patch_right='outlet')
    outlet_inner = ring(x_start, x_end, d_s/2, d_r/2, patch_inner='rotor', patch_right='outlet', patch_left='rotor')

    for i in range(n_segments):
        outlet_inner[i].set_cell_size(1, cell_size_radial)

        outlet_inner[i].n_cells[0] = int(outlet_inner[i].n_cells[0]/axial_dilution)
        outlet_inner[i].set_cell_size(0, cell_size_axial, inverse=True)

        outlet_outer[i].n_cells[0] = int(outlet_outer[i].n_cells[0]/axial_dilution)
        outlet_outer[i].set_cell_size(0, cell_size_axial, inverse=True)

    mesh.write('util/blockMeshDict.template', 'examples/meshCase/system/blockMeshDict')

    # run blockMesh
    os.system("blockMesh -case examples/meshCase")