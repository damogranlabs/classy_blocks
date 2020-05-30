from util import geometry as g
from util import constants, tools

from classes.primitives import Point, Vertex, Edge
from classes.block import Block

class Mesh():
    """ contains blocks, edges and all necessary methods for assembling blockMeshDict """
    def __init__(self):
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
                dl = g.norm(v.point.coordinates - point)
                if dl < constants.tol:
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

        tools.template_to_dict(template_path, output_path, context)