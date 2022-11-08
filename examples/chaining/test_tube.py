from classy_blocks import Cylinder, Frustum, Hemisphere, ExtrudedRing, Mesh

def get_mesh():
    # a test tube as a reactor with a part of atmosphere above and below it
    outer_diameter = 0.015 # main body,
    body_length = 0.1
    cap_diameter = 0.007 # connected to a round end cap
    conical_length = 0.02 # with a frustum of this length
    wall_thickness = 0.001

    # there's some reacting stuff in the tube, reaching up to the top of
    # the conical section
    stuff_zone = 'stuff'
    
    # outside atmosphere extends upwards, away and below the 'body'
    atm_height = 0.1
    atm_radius = 0.1

    # refer to other examples for a more proper-ish grading setup
    h = 0.001
    h_atm = 10*h

    mesh = Mesh()

    # the inside of test tube is modeled by 3 shapes
    body = Cylinder(
        [0, 0, 0],
        [0, 0, -body_length],
        [outer_diameter/2, 0, 0]
    )
    body.chop_axial(start_size=h)
    body.chop_radial(start_size=h)
    body.chop_tangential(start_size=h)
    mesh.add(body)

    cone = Frustum.chain(body, conical_length, cap_diameter/2)
    cone.chop_axial(start_size=h)
    cone.set_cell_zone(stuff_zone)
    mesh.add(cone)

    end_cap = Hemisphere.chain(cone)
    end_cap.chop_axial(start_size=h/2)
    end_cap.set_cell_zone(stuff_zone)
    mesh.add(end_cap)

    # atmosphere
    atm_above = Cylinder.chain(body, -atm_height)
    atm_above.chop_axial(start_size=h_atm)
    atm_above.set_top_patch('atmosphere')
    mesh.add(atm_above)

    atm_wall = ExtrudedRing.expand(atm_above, wall_thickness)
    atm_wall.chop_radial(start_size=h_atm)
    atm_wall.set_top_patch('atmosphere')
    mesh.add(atm_wall)

    atm_side_above = ExtrudedRing.expand(atm_wall, atm_radius)
    atm_side_above.chop_radial(start_size=h_atm)
    atm_side_above.set_top_patch('atmosphere')
    atm_side_above.set_outer_patch('atmosphere')
    mesh.add(atm_side_above)

    atm_side_below = ExtrudedRing.chain(atm_side_above, -body_length)
    atm_side_below.chop_axial(start_size=h_atm)
    atm_side_below.set_outer_patch('atmosphere')
    atm_side_below.set_top_patch('atmosphere')
    mesh.add(atm_side_below)

    mesh.set_default_patch('tube_wall', 'wall')
    
    return mesh
