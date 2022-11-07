import os

from classy_blocks.process.mesh import Mesh
from classy_blocks.construct.shapes import Hemisphere

def get_mesh():
    center_point = [0, 0, 0]
    radius_point = [0, 0, 1]
    normal = [0, 1, 0]

    cell_size = 0.1
    bl_thickness = 0.01
    
    hemisphere = Hemisphere(center_point, radius_point, normal)
    hemisphere.chop_axial(start_size=cell_size)
    hemisphere.chop_radial(start_size=cell_size, end_size=bl_thickness)
    hemisphere.chop_tangential(start_size=cell_size)

    hemisphere.set_bottom_patch('atmosphere')
    hemisphere.set_outer_patch('walls') # the same as set_top_patch


    mesh = Mesh()
    mesh.add(hemisphere)
    
    return mesh
