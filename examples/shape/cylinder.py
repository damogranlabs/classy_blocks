from classy_blocks import Cylinder, Mesh

axis_point_1 = [0, 0, 0]
axis_point_2 = [5, 5, 0]
radius_point_1 = [0, 0, 2]

def get_mesh():
    cylinder = Cylinder(axis_point_1, axis_point_2, radius_point_1)

    cylinder.set_bottom_patch('inlet')
    cylinder.set_top_patch('outlet')
    cylinder.set_outer_patch('walls')

    bl_thickness=0.05
    core_size = 0.2

    cylinder.chop_axial(count=30)
    cylinder.chop_radial(start_size=core_size, end_size=bl_thickness)
    cylinder.chop_tangential(start_size=core_size)

    mesh = Mesh()
    mesh.add(cylinder)

    return mesh
