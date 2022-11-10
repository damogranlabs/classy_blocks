from classy_blocks import FrustumWall, Mesh

def get_mesh():
    axis_point_1 = [0, 0, 0]
    axis_point_2 = [2, 2, 0]
    outer_radius_point_1 = [0, 0, 2]
    thickness_1 = 0.1

    outer_radius_2 = 0.5
    thickness_2 = 0.2

    bl_thickness = 0.005
    core_size = 0.05

    # A note about radius_mid;
    # it can be used to create shapes of revolution with curved sides;
    # however, due to the way blockMesh face creation works, the result won't
    # be totally 'round'.
    # Also, in drastic cases, non-orthogonality at beginning/end face will be high
    # because of sharp edges; in those cases it is better to use RevolvedRing combined with
    # Cylinder/Frustum with non-flat start/end faces.
    frustum = FrustumWall(
        axis_point_1, axis_point_2,
        outer_radius_point_1, thickness_1,
        outer_radius_2, thickness_2,
        outer_radius_mid=1.1,
        n_segments=12)

    frustum.set_bottom_patch('inlet')
    frustum.set_top_patch('outlet')
    frustum.set_inner_patch('walls')
    frustum.set_outer_patch('walls')

    frustum.chop_axial(count=30)
    frustum.chop_radial(start_size=core_size, end_size=bl_thickness)
    frustum.chop_tangential(start_size=core_size)

    mesh = Mesh()
    mesh.add(frustum)

    return mesh
