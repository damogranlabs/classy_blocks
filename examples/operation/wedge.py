from classy_blocks import Face, Wedge, Mesh

def get_mesh():
    mesh = Mesh()

    # a face with a single bump;
    base = Face([[0, 0, 0], [1, 0, 0], [1, 0.2, 0], [0, 0.2, 0]])
    base.add_edge(2, [
        [0.75, 0.15, 0], # a spline edge
        [0.50, 0.20, 0], # with 3
        [0.25, 0.25, 0], # points
    ], kind='spline')

    # move it away from the axis of rotation
    # x axis = [1, 0, 0]
    base = base.translate([0, 1, 0])

    # then copy it along x-axis,
    # representing an annular seal with grooves
    wedges = []
    for _ in range(5):
        wedge = Wedge(base)

        # this has to be set on all blocks;
        # others will be copied
        wedge.chop(0, count=30)

        wedge.set_outer_patch('static_wall')
        wedge.set_inner_patch('rotating_walls')

        wedges.append(wedge)
        mesh.add(wedge)

        base = base.translate([1, 0, 0])

    wedges[0].set_left_patch('inlet')
    wedges[-1].set_right_patch('outlet')

    # this will be copied to all next blocks
    wedges[0].chop(1, c2c_expansion=1.2, start_size=0.01, invert=True)

    return mesh
