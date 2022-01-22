import numpy as np

from classy_blocks.classes import mesh
from classy_blocks.classes.operations import Face, Loft, Extrude, Revolve

import classy_blocks.util.functions as f

class Geometry:
    """ calculation of a centrifugal pump is done separately;
    since that is not the topic of this showcase it is conveniently omitted.
    
    This object only holds the results that are needed for construction
    of a volute with a rectangular cross-section.   
    The actual shape is defined by four curves through points, stored in 
    this directory. See pump_volute.svg for a graphical explanation; """

    # inlet properties
    r_1T = 12 # eye radius
    r_1H = 6.5 # hub radius
    b_1 = 8.4 # channel height
    s = 1.5 # blade thickness

    # outlet properties
    r_2T = 21 # outlet radius
    b_2 = 6.5 # outlet height

    # volute properties
    r_3 = 28 # cutwater diameter
    b_3 = 12 # inlet height
    A_3q = 650 # area of throat cross-section
    e_3 = 3 # cutwater thickness

    # a ring for Arbitrary Mesh Interface between
    # impeller and volute
    p_0 = f.vector(6.15547, 21.3137, 0)
    p_1 = f.vector(5.84174, 22.472, 0)
    p_2 = f.vector(15.6741, 19.8937, 0)
    p_3 = f.vector(15.3603, 18.7355, 0)
    
g = Geometry()

class VolutePoint:
    """ The volute, defined by HI/HO/TI/TO curves, is made from blocks that are
    Lofted from one angle to another. All other blocks (except outlet)
    are just revolutions - but made with Loft operation.

    Loft requires start, end, and optional edge points.
    This object decides whether to take fixed points and rotate them
    to mimick a Revolve operation or actually choose points from curves
    for a proper Loft operation for volute 'spiral'. """
    def __init__(self, x, y):
        # x and/or y can be specified as a fixed number or as a
        # list of points;
        # in the latter case, the coordinate will be 
        # chosen from the list closest to specified index
        self.x = np.array(x)
        self.y = np.array(y)

    def de_rotate(self, p):
        # rotate the point p around x-axis so that z-axis=0
        # calculate r in cylindrical coordinate system;
        ppoint = f.to_polar(p, axis='x') # (r, phi, z)
        # return de-rotated point and angle by which it was de-rotated
        return f.vector(ppoint[2], ppoint[0], 0), ppoint[1]
    
    def get_coords(self, index, to):
        _, angle = self.de_rotate(to[index])

        if len(self.x.shape) == 2:
            # self.x is a list of points (one of ti/to/ho/hi);
            # use the x-value of the point at 'index'
            p = self.x[index]
            x = f.rotate(p, -angle)[0]
        else:
            x = self.x
        
        if len(self.y.shape) == 2:
            p = self.y[index]
            y = f.rotate(p, -angle)[1]
        else:
            y = self.y

        p = f.vector(x, y, 0)
        p = f.rotate(p, angle, axis='x')

        return p, angle

def get_mesh():
    ###
    ### the mesh
    ### 
    # load calculated points from files
    ti = np.loadtxt("examples/complex/Volute_TI.pts")
    to = np.loadtxt("examples/complex/Volute_TO.pts")
    hi = np.loadtxt("examples/complex/Volute_HI.pts")
    ho = np.loadtxt("examples/complex/Volute_HO.pts")

    # dimensions of throat cross-section
    ht = ho[-1][0] - to[-1][0]
    at = to[-1][1] - ti[-1][1]

    # assign points for revolution
    # refer to documentation/volute.svg for graphical representation
    # the points are in cylindrical coordinate system p=(x, r) and
    # will be rotated later by VolutePoint
    p = [None]*24

    p[0] = VolutePoint(-g.b_1, g.r_1T + g.s)
    p[1] = VolutePoint(p[0].x, g.p_0[1])
    p[2] = VolutePoint(p[0].x/2, p[0].y)
    p[3] = VolutePoint((g.p_0[0] + p[1].x)/2, p[1].y)

    p[6] = VolutePoint(0, p[0].y)
    p[7] = VolutePoint(g.p_0[0], g.p_0[1])
    p[8] = VolutePoint(g.p_1[0], g.p_1[1])

    p[4] = VolutePoint(p[3].x, p[8].y)
    p[5] = VolutePoint(p[4].x, g.r_3)

    p[9] = VolutePoint(hi[0][0] - g.b_3, g.r_3)
    p[10] = VolutePoint(p[9].x, ti)
    p[11] = VolutePoint(p[10].x, to)

    p[13] = VolutePoint(g.p_3[0], g.p_3[1])
    p[14] = VolutePoint(g.p_2[0], g.p_2[1])

    p[12] = VolutePoint(p[13].x, g.r_1H)

    p[15] = VolutePoint(hi[0][0], p[9].y)
    p[16] = VolutePoint(hi, hi)
    p[17] = VolutePoint(ho, ho)
    p[18] = VolutePoint(p[12].x + g.b_2 + g.s, p[12].y)
    p[19] = VolutePoint(p[18].x, p[13].y)
    p[20] = VolutePoint(p[18].x, p[14].y)
    p[21] = VolutePoint(p[18].x, p[15].y)

    p[22] = VolutePoint(ti, ti)
    p[23] = VolutePoint(to, to)

    # mesh handling
    volute = mesh.Mesh()

    # choose the indexes from points calculated in Volute()
    n_blocks = 8 # number of blocks circumferentially
    n_points = len(ti)
    # index of the point where a rectangular volute becomes square; from here on,
    # all following cross-sections increase in side axially and radially.
    # this comes from volute calculation but is hard-coded here
    i_square = 25

    # rectangular area: proportional number of blocks up to the point where x-section becomes square
    def dilute_indexes(n, l):
        """ choose <l> points from an array of length <n> """
        return np.round(np.linspace(0, n-1, l)).astype(int)

    # number of rectangular blocks
    n_rect_blocks = int(n_blocks*i_square/n_points) + 1
    rect_indexes = dilute_indexes(i_square, 2*n_rect_blocks+1)

    # square area: both blocks, 6 and 7, the rest of indexes
    n_square_blocks = n_blocks - n_rect_blocks
    square_indexes = dilute_indexes(n_points-i_square, 2*n_square_blocks+1) + rect_indexes[-1] + 1
    indexes = list(rect_indexes) + list(square_indexes[1:])

    # cell sizing:
    # block.count_to_size takes into account block size;
    # that's a problem with blocks that change dimensions but should have the same count
    cell_size = g.s

    def count_to_size(length):
        c = int(length / cell_size)
        return max(1, c)

    counts = {
        2: count_to_size(f.norm(g.p_3 - g.p_2)),
        8: count_to_size(g.r_3 - g.p_2[1])
    }

    def create_block(i_start, i_end, i_edge, p0, p1, p2, p3, description):
        # get points either by choosing from ti/to/ho/hi or by
        # revolving fixed points to the appropriate plane (the current one)
        start_points = [p.get_coords(i_start, to)[0] for p in [p0, p1, p2, p3]]
        edge_points = [p.get_coords(i_edge, to)[0] for p in [p0, p1, p2, p3]]
        end_points = [p.get_coords(i_end, to)[0] for p in [p0, p1, p2, p3]]

        start_face = Face(start_points, edges=None, check_coplanar=False)
        end_face = Face(end_points, check_coplanar=False)

        loft = Loft(start_face, end_face, edge_points)
        loft.block.description = f"(Sketch: {description})"

        volute.add(loft)

        return loft

    for i_block in range(0, n_blocks):
        i_start = indexes[2*i_block]
        i_edge = indexes[2*i_block+1]
        i_end = indexes[2*i_block+2]

        # create blocks one by one
        block_0 = create_block(i_start, i_end, i_edge, p[0], p[1], p[3], p[2], 0)
        block_0.chop(0, start_size=cell_size)
        block_0.chop(1, start_size=cell_size)
        block_0.chop(2, start_size=cell_size * g.r_1T / g.r_2T)
        block_0.set_patch('left', 'volute_rotating')

        block_1 = create_block(i_start, i_end, i_edge, p[2], p[3], p[7], p[6], 1)
        block_1.chop(1, start_size=cell_size)
        block_1.set_patch(['left', 'back'], 'volute_rotating')

        block_2 = create_block(i_start, i_end, i_edge, p[3], p[4], p[8], p[7], 2)
        block_2.chop(0, count=counts[2])
        block_2.set_patch('back', 'volute_inlet')

        block_3 = create_block(i_start, i_end, i_edge, p[4], p[5], p[9], p[8], 3)
        block_3.chop(0, count=counts[8])

        block_4 = create_block(i_start, i_end, i_edge, p[8], p[9], p[15], p[14], 4)
        block_4.set_patch('left', 'volute_inlet')
        if i_block == 0:
            block_4.chop(1, start_size=cell_size)
        
        block_5 = create_block(i_start, i_end, i_edge,  p[9], p[10], p[16], p[15], 5)
        block_5.chop(0, count=(1 + g.e_3 / cell_size))
        
        block_6 = create_block(i_start, i_end, i_edge, p[10], p[11], p[17], p[16], 6)
        block_6.chop(0, count=at/cell_size)

        if i_end > i_square:
            # add a block 7 if cross-section is big enough already
            block_7 = create_block(i_start, i_end, i_edge, p[22], p[23], p[11], p[10], 7)
            block_7.chop(1, count=(ht - g.b_3)/cell_size)
        
        block_8 = create_block(i_start, i_end, i_edge, p[14], p[15], p[21], p[20], 8)
        block_8.chop(1, start_size=cell_size)
        
        block_9 = create_block(i_start, i_end, i_edge, p[13], p[14], p[20], p[19], 9)
        block_9.chop(0, count=counts[2])
        block_9.set_patch('front', 'volute_inlet')

        block_10 = create_block(i_start, i_end, i_edge, p[12], p[13], p[19], p[18], 10)
        block_10.chop(0, start_size=cell_size)
        block_10.set_patch(['left', 'front'], 'volute_rotating')


    ### radial discharge curve consists of 2 sets of two blocks:
    # loft (radius) and outlet pipe, each set from a separate volute block, 6 and 7
    def discharge_set(start_face):
        pti = start_face.points[0]
        pto = start_face.points[1]
        pho = start_face.points[2]
        phi = start_face.points[3]

        # recommended radius of curvature of discharge pipe
        # (according to J. F. Gulich, Centrifugal Pumps)
        r_n = 1.5*np.sqrt(4*g.A_3q/np.pi)

        # find the point around which the rectangle above will be revolved
        inner_mid = phi + (pti - phi)/2
        outer_mid = pho + (pto - pho)/2
        a_mid = outer_mid - inner_mid

        p_rev = outer_mid + a_mid/f.norm(a_mid) * (r_n - f.norm(a_mid)/2)
        
        # revolve the four end_* points around p_rev
        eps0 = f.deg2rad(30)
        angle = -(eps0 - np.pi/2)

        discharge_revolve = Revolve(start_face, angle, [1, 0, 0], p_rev)
        discharge_revolve.chop(2, start_size=cell_size)

        volute.add(discharge_revolve)

        # outlet pipe: 'extrude' the last face by 5*(pipe diameter)
        # find the direction of extrude first
        discharge_end_face = discharge_revolve.top_face
        v_extrude = f.rotate(
            discharge_end_face.points[1] - discharge_end_face.points[0],
            -np.pi/2, axis='x')
        v_extrude = v_extrude/f.norm(v_extrude)*at*5

        outlet_extrude = Extrude(discharge_end_face, v_extrude)
        outlet_extrude.set_patch('top', 'volute_outlet')

        # here we can afford bigger cells since the this part is not that important
        outlet_extrude.chop(2, start_size=cell_size, end_size=cell_size*3)

        volute.add(outlet_extrude)

    discharge_set(block_6.top_face)
    discharge_set(block_7.top_face)


    # what's left to do to get a fully working mesh:
    #  - createPatch to create "walls" from undefined "defaultFaces"
    #  - collapseEdges to get rid of zero-sized faces on cutwater
    #  - refineWallLayer, refineHexMesh, whatever additional polish is needed
    #  - checkMesh to make sure it works well

    # Another option: use this mesh (which is not ideal) to obtain STL files
    # for another meshing tool (snappyHexMesh, cfMesh, ...) and generate a 
    # also-not-ideal mesh there: https://damogranlabs.com/2020/10/blockmesh-for-external-flows/
    # tl;dr:
    #  - blockMesh
    #  - surfaceMeshExtract
    #  - Optionally: surfaceRefineRedGreen, surfaceLambdaMuSmooth
    #  - snappyHexMesh/cfMesh
    return volute
