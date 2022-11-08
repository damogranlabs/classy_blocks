import os
import numpy as np

from classy_blocks import Face, Extrude, Mesh

def load_airfoil_file(filename, chord=1):
    points_upper = []
    points_lower = []

    def line_to_numbers(line):
        line = line.strip()
        p2d = [float(s) for s in line.split()]

        # add a z-coordinate
        return [p2d[0], p2d[1], 0]

    with open(filename, 'r') as f:
        # reads Lednicer airfoil file from airfoiltools.com
        # sample: http://airfoiltools.com/airfoil/details?airfoil=tempest1-il

        # first line: name, nothing useful
        f.readline()

        # second line: number of points for upper and lower portion
        n_points = line_to_numbers(f.readline())

        # an empty line
        f.readline()

        for _ in range(int(n_points[0])):
            points_upper.append(line_to_numbers(f.readline()))

        # an empty line between upper and lower part
        f.readline()

        for _ in range(int(n_points[1])):
            points_lower.append(line_to_numbers(f.readline()))
    
    return np.array(points_upper)*chord, np.array(points_lower)*chord

# finding points closest to wanted coordinate
def find_y(points, x):
    i_result = 0
    y_result = 0

    for i, p in enumerate(points):
        if p[0] > x:
            i_result = i
            y_result = p[1]
            break

    return i_result, y_result

def get_mesh():
    ###
    ### parameters
    ###
    chord = 0.5 # chord length [m]
    domain_radius = 1.0 # defines domain size in front of the airfoil and domain height [m]
    radius_center = 0.30 # position of circle radius, relative to chord []

    domain_length = 10.0*chord # length of rectangular section behind the half-circle [m]
    thickness = [0, 0, 0.2] # domain thickness (extrude vector)

    cell_size = 0.01

    ###
    ### point preparation
    ###
    p_upper, p_lower = load_airfoil_file(os.path.join('examples', 'operation', 'airfoil_1.dat'), chord=chord)

    ###
    ### block creation
    ###

    # top block 1
    i, y, = find_y(p_upper, chord*radius_center)
    max_x_1 = chord*radius_center
    radius_edge = chord*domain_radius*2**0.5

    face_top_1_vertices = [
        [max_x_1-domain_radius, 0, 0],
        [0, 0, 0],
        [max_x_1, y, 0],
        [max_x_1, domain_radius, 0],
    ]
    face_top_1_edges = [
        None,
        p_upper[0:i-1],
        None,
        [-radius_edge + chord*radius_center, radius_edge, 0]
    ]

    # create a face from points and edges
    face_top_1 = Face(face_top_1_vertices, face_top_1_edges)
    # create an Extrude operation from face and extrude vector
    extrude_top_1 = Extrude(face_top_1, thickness)

    # set cell counts on all axes for the first block
    extrude_top_1.chop(0, start_size=cell_size, c2c_expansion=1.1, invert=True)
    extrude_top_1.chop(1, count=30)
    extrude_top_1.chop(2, count=1)

    # top block 2
    face_top_2_vertices = [
        face_top_1_vertices[2],
        [chord, 0, 0],
        [chord, domain_radius, 0],
        [max_x_1, domain_radius, 0]
    ]

    face_top_2_edges = [p_upper[i:], None, None, None]

    face_top_2 = Face(face_top_2_vertices, face_top_2_edges)
    extrude_top_2 = Extrude(face_top_2, thickness)
    extrude_top_2.chop(0, start_size=cell_size)
    # other cell counts must match other blocks' so they need not be set

    # top block 3
    face_top_3 = Face([
        [chord, 0, 0],
        [domain_length, 0, 0],
        [domain_length, domain_radius, 0],
        [chord, domain_radius, 0]
    ])
    extrude_top_3 = Extrude(face_top_3, thickness)
    extrude_top_3.chop(0, start_size=cell_size, c2c_expansion=1.1)

    # bottom block 1
    i, y, = find_y(p_lower, chord*radius_center)

    face_bottom_1_vertices = [
        [max_x_1-domain_radius, 0, 0],
        [max_x_1, -domain_radius, 0],
        [max_x_1, y, 0],
        [0, 0, 0],
    ]
    face_bottom_1_edges = [
        [-radius_edge + chord*radius_center, -radius_edge, 0],
        None,
        np.flip(p_lower[0:i-1], axis=0), # this block is defined in reverse so edge points must be reversed as well
        None
    ]

    face_bottom_1 = Face(face_bottom_1_vertices, face_bottom_1_edges)
    extrude_bottom_1 = Extrude(face_bottom_1, thickness)
    extrude_bottom_1.chop(0, count=30)

    # bottom block 2
    face_bottom_2_vertices = [
        face_bottom_1_vertices[2],
        [max_x_1, -domain_radius, 0],
        [chord, -domain_radius, 0],
        [chord, 0, 0]
    ]

    face_bottom_2_edges = [None, None, None, np.flip(p_lower[i:], axis=0)]

    face_bottom_2 = Face(face_bottom_2_vertices, face_bottom_2_edges)
    extrude_bottom_2 = Extrude(face_bottom_2, thickness)
    extrude_bottom_2.chop(1, start_size=cell_size)

    # bottom block 3
    face_bottom_3 = Face([
        [chord, 0, 0],
        [chord, -domain_radius, 0],
        [domain_length, -domain_radius, 0],
        [domain_length, 0, 0]
    ])
    extrude_bottom_3 = Extrude(face_bottom_3, thickness)


    mesh = Mesh()
    mesh.add(extrude_top_1)
    mesh.add(extrude_top_2)
    mesh.add(extrude_top_3)

    mesh.add(extrude_bottom_1)
    mesh.add(extrude_bottom_2)
    mesh.add(extrude_bottom_3)

    return mesh
