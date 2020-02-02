# -*- coding: utf-8 -*-
import os, jinja2
import numpy as np
import scipy.optimize

import functions as f

# geometric tolerance
tol = 1e-7

# mesh utils
def template_to_dict(template_path, dict_path, context):
    """ renders template with context to product a dictionary (or anything else) """
    template_file = open(template_path, "r")
    template_text = template_file.read()
    template_file.close()

    template = jinja2.Template(template_text)
    mesh_file = open(dict_path, "w")
    mesh_file.write(template.render(context))
    mesh_file.close()

# see README for terminology, terminolology, lol
class Point():
    """ vertex without an index (for, like, edges and stuff) """
    def __init__(self, coordinates):
        self.coordinates = coordinates
        
    @property
    def short_output(self):
        # output for edge definitions
        return "({} {} {})".format(self.coordinates[0], self.coordinates[1], self.coordinates[2]) 
    
    @property
    def long_output(self):
        # output for Vertex definitions
        return "({}\t{}\t{})".format(
            self.coordinates[0], self.coordinates[1], self.coordinates[2]
        )

    def __add__(self, other):
        return np.array(self.coordinates) + np.array(other.coordinates)

    def __sub__(self, other):
        return np.array(self.coordinates) - np.array(other.coordinates)


class Vertex():
    """ point with an index that's used in block and face definition """
    def __init__(self, point, index):
        self.point = Point(point)
        self.index = index
        
    @property
    def output(self):
        return self.point.long_output + " // {}".format(self.index)
    
    @property
    def printout(self):
        return "Vertex {} {}".format(self.index, self.point.short_output)
    
    def __repr__(self):
        return self.output

    def __str__(self):
        return self.output
        
    def __unicode__(self):
        return self.output


class Edge():
    def __init__(self, vertex_1, vertex_2, point, index):
        """ an edge is defined by two vertices and a point in between;
            all edges are treated as circular arcs
        """
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2
        self.point = Point(point)

    @property
    def output(self):
        return "arc {} {} {}".format(
            self.vertex_1.index,
            self.vertex_2.index,
            self.point.short_output
        )

    @property
    def is_valid(self):
        # returns True if the edge is not an arc;
        # in case the three points are collinear, blockMesh
        # will find an arc with infinite radius and crash.
        OA = np.array(self.vertex_1.point.coordinates)
        OB = np.array(self.vertex_2.point.coordinates)
        OC = np.array(self.point.coordinates)

        # if point C is on the same line as A and B:
        # OC = OA + k*(OB-OA)
        AB = OB - OA
        AC = OC - OA

        k = f.norm(AC)/f.norm(AB)
        d = f.norm((OA+AC) - (OA + k*AB))

        return d > tol

    def __repr__(self):
        return self.output


class Block():
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
            return f.norm(
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
        H = scipy.optimize.newton(fL, 1)
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

class Mesh():
    def __init__(self):
        """ contains blocks and methods for assembling blockMeshDict """
        self.vertices = [] # list of vertices
        self.edges = [] # list of edges
        self.blocks = [] # list of blocks


    @property
    def iVertex(self):
        return len(self.vertices)

    @property
    def iEdge(self):
        return len(self.edges)
    
    @property
    def iBlock(self):
        return len(self.blocks)

    def add_vertex(self, point, duplicate=False):
        """ creates a vertex, adds it to the mesh and returns the Vertex() object"""
        vertex = None

        if not duplicate:
            # check if there's already a vertex at this point;
            # if so, just return the existing vertex
            for v in self.vertices:
                dl = f.norm(v.point.coordinates - point)
                if dl < tol:
                    return v

        # create a new vertex even if there's already one in the same place
        vertex = Vertex(point, self.iVertex)
        self.vertices.append(vertex)

        return vertex

    def add_vertices(self, points, duplicate=False):
        """ the same as add_vertex but returns a list of Vertex objects """
        return [self.add_vertex(p, duplicate=duplicate) for p in points]

    def add_edge(self, v_1, v_2, point):
        # check if there's the same edge in the list already;
        # if there is, just return a reference to that
        for e in self.edges:
            if set([v_1.index, v_2.index]) == set([e.vertex_1.index, e.vertex_2.index]):
                # print("Duplicated edge: {}".format(e))
                return e

        edge = Edge(v_1, v_2, point, self.iEdge)
        if edge.is_valid:
            self.edges.append(edge)
            return True
        else:
            del edge
            return False

    def add_block(self, vertices, cells, cellZone=None, description=""):
        block = Block(vertices, self.iBlock, cells, cellZone, description)
        self.blocks.append(block)

        return block

    @property
    def patches(self):
        # Block contains patches according to the example in __init__()
        # this method collects all faces for a patch name from all blocks

        # collect all patch names
        patch_names = []
        for block in self.blocks:
            patch_names += list(block.patches.keys())

        # keep the patches unique
        patch_names = list(set(patch_names))

        # create a dict with all patches
        patches = { name: [] for name in patch_names }

        # gather all faces of all blocks
        for block in self.blocks:
            for patch_name in patch_names:
                patches[patch_name] += block.get_faces(patch_name)
        
        return patches

    def write(self, template_path, output_path, context=None):
        context = {
            'vertices': self.vertices,
            'edges': self.edges,
            'blocks': self.blocks,
            'patches': self.patches,
        }

        template_to_dict(template_path, output_path, context)