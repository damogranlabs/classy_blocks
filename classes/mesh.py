import os
import numpy as np

from ..util import functions as g
from ..util import constants, tools

class Mesh():
    """ contains blocks, edges and all necessary methods for assembling blockMeshDict """
    def __init__(self):
        self.vertices = [] # list of vertices
        self.edges = [] # list of edges
        self.blocks = [] # list of blocks
        self.faces = [] # projected faces

        self.default_patch = None

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

    def add(self, item):
        if hasattr(item, 'block'):
            self.add_block(item.block)
        else:
            for block in item.blocks:
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

    def assign_neighbours(self, block):
        # TEST
        for axis in range(3):
            axis_pairs = block.get_axis_vertex_pairs(axis)

            for i in range(len(self.blocks)):
                mb = self.blocks[i]

                for p in axis_pairs:
                    b_axis, _ = mb.get_axis_from_pair(p)
                    if b_axis is not None:
                        # block 'mb' shares the same edge or face
                        block.neighbours.add(mb)


    def copy_count(self, block, axis):
        """ finds a block that shares an edge with given block
        and copies its cell count along that edge """
        # there are 4 pairs of vertices on specified axis:
        match_pairs = block.get_axis_vertex_pairs(axis)

        # first, find a block in mesh that shares one of the
        # edges in match_pairs:
        for b in block.neighbours:
            for p in match_pairs:
                b_axis, _ = b.get_axis_from_pair(p)
                if b_axis is not None:
                    # b.get_axis_from_pair() returns axis index in
                    # the block we want to copy from;
                    if b.n_cells[b_axis] is not None:
                        # this block has the cell count set
                        # so we can (must) copy it
                        block.n_cells[axis] = b.n_cells[b_axis]
                        return True
        
        return False

    def copy_grading(self, block, axis):
        """ same as self.copy_count but for grading """
        match_pairs = block.get_axis_vertex_pairs(axis)

        for b in block.neighbours:
            for p in match_pairs:
                b_axis, direction = b.get_axis_from_pair(p)
                if b_axis is not None:
                    # b.get_axis_from_pair() returns axis index in
                    # the block we want to copy from
                    if b.grading[b_axis] is not None:
                        # this block has the cell count set;
                        # if it's created in reverse, invert the grading as well
                        if direction:
                            block.grading[axis] = b.grading[b_axis].copy()
                        else:
                            block.grading[axis] = b.grading[b_axis].copy(invert=True)

                        return True

        return False

    def prepare_data(self):
        # collect all vertices from all blocks,
        # check for duplicates and give them indexes
        for block in self.blocks:
            for i, block_vertex in enumerate(block.vertices):
                found_vertex = self.find_vertex(block_vertex)

                if found_vertex is None:
                    block.vertices[i].mesh_index = len(self.vertices)
                    self.vertices.append(block_vertex)
                else:
                    block.vertices[i] = found_vertex
        
        # collect all edges from all blocks;
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

            # projected edges:
            for e in block.projected_edges:
                # TEST
                # no special checking here (TODO?)
                v_1 = block.vertices[e[0]].mesh_index
                v_2 = block.vertices[e[1]].mesh_index

                self.edges.append(f"project {v_1} {v_2} ({e[2]})")
                

        # collect all neighbours from all blocks;
        # when setting counts and gradings, each block will iterate over them
        # only and not through all blocks
        for block in self.blocks:
            self.assign_neighbours(block)

        # now is the time to set counts
        for block in self.blocks:
            for f in block.deferred_counts:
                f.call()

        # propagate cell count:
        # a riddle similar to sudoku, keep traversing
        # and copying counts until there's no undefined blocks left
        n_blocks = len(self.blocks)
        undefined_blocks = set(range(n_blocks))
        updated = False

        while len(undefined_blocks) > 0:
            for i in undefined_blocks:
                block = self.blocks[i]

                for axis in range(3):
                    if block.n_cells[axis] is None:
                        updated = self.copy_count(block, axis) or updated
                
                if block.is_count_defined:
                    undefined_blocks.remove(i)
                    updated = True
                    break
                        
            if not updated:
                # a whole iteration went around without an update;
                # next iterations won't get any better
                break

            updated = False
        
        if len(undefined_blocks) > 0:
            # gather more detailed information about non-defined blocks:
            message = "Blocks with non-defined counts: \n"
            for i in list(undefined_blocks):
                message += f"\t{i}: {str(self.blocks[i].n_cells)}\n"
            message += '\n'
            
            raise Exception(message)
        
        # now is the time to set gradings
        for block in self.blocks:
            for f in block.deferred_gradings:
                f.call()

        # propagate grading:
        # very similar to counts
        undefined_blocks = set(range(n_blocks))
        updated = False

        while len(undefined_blocks) > 0:
            for i in undefined_blocks:
                block = self.blocks[i]

                for axis in range(3):
                    if block.grading[axis] is None:
                        updated = self.copy_grading(block, axis) or updated

                if block.is_grading_defined:
                    undefined_blocks.remove(i)
                    updated = True
                    break

            if not updated:
                break

            updated = False
        
        # projected faces:
        for b in self.blocks:
            for f in b.projected_faces:
                self.faces.append([
                    b.format_face(f[0]), # face, like (8 12 15 11) 
                    f[1] # the geometry to project to
                ])

    def set_default_patch(self, name, type):
        assert type in ('patch', 'wall', 'empty', 'wedge')

        self.default_patch = {
            'name': name,
            'type': type
        }

    def write(self, output_path, context=None, template_path=None):
        # if template path is not given, find the default relative to this file
        if template_path is None:
            classy_dir = os.path.dirname(__file__)
            template_path = os.path.join(classy_dir, '..', 'util', 'blockMeshDict.template')

        self.prepare_data()

        context = {
            'vertices': self.vertices,
            'edges': self.edges,
            'blocks': self.blocks,
            'patches': self.patches,
            'faces': self.faces,
            'default_patch': self.default_patch,
        }

        tools.template_to_dict(template_path, output_path, context)
