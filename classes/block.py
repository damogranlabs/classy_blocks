import numpy as np
import scipy.optimize

from util import geometry as g

class Block():
    """ a direct representation of a blockMesh block; contains all necessary data to create it """
    def __init__(self, vertices, index, n_cells, cellZone=None, description=""):
        # a list of 8 Vertex object for each corner of the block
        self.vertices = vertices

        # block sequence
        self.index = index

        # a list of number of cells for each direction (length 3)
        assert len(n_cells) == 3
        self.n_cells = n_cells

        # cellZone to which the block belongs to
        self.cellZone = "" if cellZone is None else cellZone

        # written as a comment after block description
        self.description = description

        # block grading, can be modified later
        self.grading = [1, 1, 1]

        # connection between sides (top/left, ...) and faces (vertex indexes)
        self.face_map = {
            'top': (4, 5, 1, 0), 
            'bottom': (7, 6, 2, 3),
            'left': (4, 5, 6, 7),
            'right': (0, 1, 2, 3),
            'front': (4, 0, 3, 7),
            'back': (5, 1, 2, 6),
        }

        # patches: an example
        # self.patches = {
        #     'volute_rotating': ['left', 'top' ],
        #     'volute_walls': ['bottom'],
        # }
        self.patches = { }


    def set_patch(self, sides, patch_name):
        # see patches: an example in __init__()
        if patch_name not in self.patches:
            self.patches[patch_name] = []
        
        self.patches[patch_name] += sides

    def get_faces(self, patch):
        if patch not in self.patches:
            return []

        sides = self.patches[patch]
        faces = []

        for side in sides:
            faces.append(self.format_face(side))
        
        return faces

    def format_face(self, side):
        indexes = self.face_map[side]
        vertices = np.take(self.vertices, indexes)

        return "({} {} {} {})".format(
            vertices[0].index,
            vertices[1].index,
            vertices[2].index,
            vertices[3].index
        )       

    @property
    def definition(self):
        """ outputs block's definition for blockMeshDict file """
        # hex definition
        output = "hex "
        # vertices
        output += " ( " + " ".join(str(v.index) for v in self.vertices) + " ) "
    
        # cellZone
        output += self.cellZone
        # number of cells
        output += " ({} {} {}) ".format(self.n_cells[0], self.n_cells[1], self.n_cells[2])
        # grading
        output += " simpleGrading ({} {} {})".format(self.grading[0], self.grading[1], self.grading[2])

        # add a comment with block index
        output += " // {} {}".format(self.index, self.description)

        return output

    @property
    def size(self):
        # returns an approximate block dimensions:
        # for the sake of simplicity and common sense
        # replace curved edges by straignt ones

        # x-direction: vertices 0 and 1
        # y-direction: vertices 1 and 2
        # z-direction: vertices 0 and 4
        def vertex_distance(index_1, index_2):
            return g.norm(
                self.vertices[index_1].point.coordinates - self.vertices[index_2].point.coordinates
            )
        
        return [
            vertex_distance(0, 1),
            vertex_distance(1, 2),
            vertex_distance(0, 4),
        ]
    
    def set_cell_size(self, axis, cell_size, inverse=False):
        # geometric series with known sum and ratio between first and last term
        # https://en.wikipedia.org/wiki/Geometric_series#Formula
        n = self.n_cells[axis]
        L = self.size[axis]

        # sum of all terms
        def fL(h):
            if h == 1:
                return n*cell_size - L
            else:
                return cell_size*((1-h**n) / (1-h)) - L 

        # H - common ratio
        # H = scipy.optimize.newton(fL, 1) # newton's method sometimes finds zero too soon
        # TODO: improve + test + improve again + test again
        H = scipy.optimize.brentq(fL, 0.1, 10)
        # G - block grading
        G = 1/H**n

        # setting first or last cell size?
        if inverse:
            G = 1/G
        
        self.grading[axis] = G

    def __str__(self):
        return self.definition
    def __repr__(self):
        return self.definition
    def __unicode__(self):
        return self.definition