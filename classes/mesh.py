import numpy as np

from util import functions as g
from util import constants, tools

from classes.primitives import Vertex, Edge
from classes.block import Block

class Mesh():
    """ contains blocks, edges and all necessary methods for assembling blockMeshDict """
    def __init__(self):
        self.vertices = [] # list of vertices
        self.edges = [] # list of edges
        self.blocks = [] # list of blocks

    def find_vertex(self, new_vertex):
        """ checks if any of existing vertices in self.vertices are
        in the same location as the passed one; if so, returns
        the existing vertex """

        for mesh_vertex in self.vertices:
            if g.norm(mesh_vertex.point - new_vertex.point) < constants.tol:
                return mesh_vertex

        return None

    def find_edge(self, vertex_1, vertex_2):
        """ checks if an edge with the same pair of vertices
        exists in self.edges already """
        for e in self.edges:
            mesh_set = set([vertex_1.mesh_index, vertex_2.mesh_index])
            edge_set = set([e.vertex_1.mesh_index, e.vertex_2.mesh_index])
            if mesh_set == edge_set:
                return e

        return None

    def add_block(self, block):
        # block.mesh_index is not required but will come in handy for debugging
        block.mesh_index = len(self.blocks)
        self.blocks.append(block)

    def add_operation(self, operation):
        self.add_block(operation.block)

    def add_shape(self, shape):
        for block in shape.blocks:
            self.add_block(block)

    @property
    def patches(self):
        # Block contains patches according to the example in __init__()
        # this method collects all faces for a patch name from all blocks
        # (a format ready for jinja2 template)

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

    def copy_cells(self, block, axis):
        """ finds a block that shares an edge with given block
        and copies its cell count along that edge """
        # there are 4 pairs of vertices on specified axis:
        match_pairs = block.get_axis_vertex_pairs(axis)

        # first, find a block in mesh that shares one of the
        # edges in match_pairs:
        for b in self.blocks:
            for p in match_pairs:
                b_axis = b.get_axis_from_pair(p)
                if b_axis is not None:
                    # b.get_axis_from_pair() returns axis index in
                    # the block we want to copy from
                    if b.n_cells[b_axis] is not None:
                        # this block has the cell count set
                        # so we can (must) copy it
                        block.n_cells[axis] = b.n_cells[b_axis]
                        block.grading[axis] = b.grading[b_axis]
                        return True

        return False

    def prepare_data(self):
        # 1. collect all vertices from all blocks,
        # check for duplicates and give them indexes
        for block in self.blocks:
            for i, block_vertex in enumerate(block.vertices):
                found_vertex = self.find_vertex(block_vertex)

                if found_vertex is None:
                    block.vertices[i].mesh_index = len(self.vertices)
                    self.vertices.append(block_vertex)
                else:
                    block.vertices[i] = found_vertex
        
        # 2. collect all edges from all blocks;
        for block in self.blocks:
            # check for duplicates (same vertex pairs) and
            # check for validity (no straight-line arcs)
            for i, block_edge in enumerate(block.edges):
                # block.vertices by now contain index to mesh.vertices;
                # edge vertex therefore refers to actual mesh vertex
                v_1 = block.vertices[block_edge.block_index_1]
                v_2 = block.vertices[block_edge.block_index_2]

                block.edges[i].vertex_1 = v_1
                block.edges[i].vertex_2 = v_2

                if not block_edge.is_valid:
                    # invalid edges should not be added
                    continue

                if self.find_edge(v_1, v_2) is None:
                    # only non-existing edges are added
                    self.edges.append(block_edge)

        # 3. run deferred block functions
        for block in self.blocks:
            for f in block.deferred_functions:
                f.call()

        # 4. propagate cell count and grading:
        # the first block needs all three cell counts defined.
        # next blocks must copy count from the previous block
        # on shared edges
        for i_block, block in enumerate(self.blocks):
            for i_axis in range(3):
                if block.n_cells[i_axis] is None:
                    # if this is the first block, we'll find no blocks to match count;
                    # this will not work.
                    if i_block == 0:
                        raise Exception("Set cell counts for all axes on the first added block!")

                if not self.copy_cells(block, i_axis):
                    message = f"Could not find a neighbouring block to copy cell count: block {i_block}, axis {i_axis}"
                    raise Exception(message)


    def write(self, template_path, output_path, context=None):
        self.prepare_data()

        context = {
            'vertices': self.vertices,
            'edges': self.edges,
            'blocks': self.blocks,
            'patches': self.patches,
        }

        tools.template_to_dict(template_path, output_path, context)