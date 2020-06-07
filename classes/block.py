import numpy as np
import scipy.optimize

from util import geometry as g
from util import constants

from classes.primitives import Vertex, Edge

class Block():
    """ a direct representation of a blockMesh block;
    contains all necessary data to create it """
    def __init__(self, vertices, edges):
        # a list of 8 Vertex object for each corner of the block
        self.vertices = vertices
        self.edges = edges

        # other block data, set to default at creation and updated later
        # a list of number of cells for each direction
        self.n_cells = [10, 10, 10]

        # cellZone to which the block belongs to
        self.cellZone = ""

        # written as a comment after block description
        self.description = ""

        # block grading
        self.grading = [1, 1, 1] 

        # a more intuitive and quicker way to set patches,
        # according to this sketch: https://www.openfoam.com/documentation/user-guide/blockMesh.php
        self.face_map = {
            'top': (4, 5, 6, 7),
            'bottom': (0, 1, 2, 3),
            'left': (4, 0, 3, 7),
            'right': (5, 1, 2, 6),
            'front': (4, 5, 1, 0), 
            'back': (7, 6, 2, 3)
        }

        # patches: an example
        # self.patches = {
        #     'volute_rotating': ['left', 'top' ],
        #     'volute_walls': ['bottom'],
        # }
        self.patches = { }

        # set in Mesh.prepare_data()
        self.mesh_index = None

    @classmethod
    def create_from_points(cls, points, edges=[]):
        """ create a block from a raw list of 8 points;
        edges are optional; edge's 2 vertex indexes refer to
        block.vertices list (0 - 7) """
        block = cls(
            [Vertex(p) for p in points],
            edges
        )

        return block

    ###
    ### information/manipulation
    ###
    def set_patch(self, sides, patch_name):
        """ assign one or more block faces (self.face_map)
        to a chosen patch name """
        # see patches: an example in __init__()

        if type(sides) == str:
            sides = [sides]

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
            vertices[0].mesh_index,
            vertices[1].mesh_index,
            vertices[2].mesh_index,
            vertices[3].mesh_index
        )       

    @property
    def size(self):
        # returns an approximate block dimensions:
        # if an edge is defined, use the edge.get_length(),
        # otherwise simply distance between two points

        # x-direction: vertices 0 and 1
        # y-direction: vertices 1 and 2
        # z-direction: vertices 0 and 4
        def find_edge(index_1, index_2):
            # TODO:
            # at the moment, edges are not aware
            # if their vertices (edge.vertex_* = None);
            # so they can't tell the length
            #for e in self.edges:
            #    if {e.block_index_1, e.block_index_2} == {index_1, index_2}:
            #        return e
            return None

        def vertex_distance(index_1, index_2):
            return g.norm(
                self.vertices[index_1].point - self.vertices[index_2].point
            )
    
        def block_size(index_1, index_2):
            # TODO: see above
            #edge = find_edge(index_1, index_2)
            #if edge:
            #    return edge.get_length()

            return vertex_distance(index_1, index_2)
        
        return [
            block_size(0, 1), # TODO: take average of all edges in this direction
            block_size(1, 2),
            block_size(0, 4),
        ]
    
    def set_cell_size(self, axis, cell_size, inverse=False):
        # calculate grading so that first cell will be of cell_size
        # for this axis.
        n_cells = self.n_cells[axis]
        block_size = self.size[axis]

        assert cell_size < block_size

        cell_sizes = []

        def fcell_size(grading):
            nonlocal cell_sizes

            cell_sizes = [1] # will be scaled later

            # returns last cell size for given grading;
            # scales everything so that block size matches the original
            l_block = 0

            for _ in range(n_cells):
                l_block += cell_sizes[-1]
                cell_sizes.append(cell_sizes[-1]*grading)

                if cell_sizes[-1] < constants.tol:
                    return
            
            cell_sizes = [c*block_size/l_block for c in cell_sizes]

            return cell_size - cell_sizes[-1]

        # find a grading that produces correct last_cell_size
        scipy.optimize.newton(fcell_size, 1)

        if inverse:
            self.grading[axis] = cell_sizes[-1]/cell_sizes[0]
        else:
            self.grading[axis] = cell_sizes[0]/cell_sizes[-1]

    def __repr__(self):
        """ outputs block's definition for blockMeshDict file """
        # hex definition
        output = "hex "
        # vertices
        output += " ( " + " ".join(str(v.mesh_index) for v in self.vertices) + " ) "
    
        # cellZone
        output += self.cellZone
        # number of cells
        output += " ({} {} {}) ".format(self.n_cells[0], self.n_cells[1], self.n_cells[2])
        # grading
        output += " simpleGrading ({} {} {})".format(self.grading[0], self.grading[1], self.grading[2])

        # add a comment with block index
        output += " // {} {}".format(self.mesh_index, self.description)

        return output