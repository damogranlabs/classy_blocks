from classy_blocks import Cylinder, Hemisphere, Mesh


def get_mesh():
    # a cylindrical tank with round end caps
    diameter = 0.5
    length = 0.5 # including end caps
    
    # refer to other examples for a proper-ish grading setup
    h = 0.05

    mesh = Mesh()

    cylinder = Cylinder(
        [0, 0, 0],
        [length, 0, 0],
        [0, diameter/2, 0]
    )

    wall_name = 'tank_wall'

    cylinder.chop_axial(start_size=h)
    cylinder.chop_radial(start_size=h)
    cylinder.chop_tangential(start_size=h)
    cylinder.set_outer_patch(wall_name)
    mesh.add(cylinder)

    start_cap = Hemisphere.chain(cylinder, start_face=True)
    start_cap.chop_axial(start_size=h)
    start_cap.set_outer_patch(wall_name)
    mesh.add(start_cap)

    end_cap = Hemisphere.chain(cylinder, start_face=False)
    end_cap.chop_axial(start_size=h)
    end_cap.set_outer_patch(wall_name)
    mesh.add(end_cap)
    
    return mesh
