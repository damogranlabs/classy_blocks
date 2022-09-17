""" Mesh object """
import os

from typing import NoReturn, Union, List

from ..util import functions as g
from ..util import constants, tools

from .primitives import Vertex, Edge
from .block import Block
from .operations import Operation
from .shapes import Shape

class Mesh():
    """ Contains blocks, edges and all necessary methods for assembling blockMeshDict """
    output_path = 'debug.vtk'

    def __init__(self):
        self.vertices = [] # list of vertices
        self.edges = [] # list of edges
        self.blocks = [] # list of blocks
        self.patches = [] # defined in get_patches()
        self.faces = [] # projected faces

        self.default_patch = None
        self.merged_patches = [] # [['master1', 'slave1'], ['master2', 'slave2']]
        self.geometry = {}

    def find_vertex(self, new_vertex:Vertex) -> Vertex:
        """ checks if any of existing vertices in self.vertices are
        in the same location as the passed one; if so, returns
        the existing vertex """

        for mesh_vertex in self.vertices:
            if g.norm(mesh_vertex.point - new_vertex.point) < constants.tol:
                return mesh_vertex

        return None

    def find_edge(self, vertex_1:Vertex, vertex_2:Vertex) -> Edge:
        """ checks if an edge with the same pair of vertices
        exists in self.edges already """
        for e in self.edges:
            mesh_set = set([vertex_1.mesh_index, vertex_2.mesh_index])
            edge_set = set([e.vertex_1.mesh_index, e.vertex_2.mesh_index])
            if mesh_set == edge_set:
                return e

        return None

    def add_block(self, block:Block) -> NoReturn:
        """ Add a low-level Block object to this mesh """
        # block.mesh_index is not required but will come in handy for debugging
        block.mesh_index = len(self.blocks)
        self.blocks.append(block)
        
    def add(self, item: Union[Block, Operation, Shape]) -> NoReturn:
        """ Add any kind of object (Block/Operation/Shape) to this mesh """
        if hasattr(item, 'block'):
            self.add_block(item.block)
        elif hasattr(item, 'blocks'):
            for block in item.blocks:
                self.add_block(block)
        else:
            self.add_block(item)

        # TODO: TEST
        if hasattr(item, 'geometry'):
            self.add_geometry(item.geometry)

    def get_patches(self) -> dict:
        """ Block contains patches according to the example in __init__()
        this method collects all faces for a patch name from all blocks
        (a format ready for jinja2 template) """

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

    def assign_neighbours(self, block:Block) -> NoReturn:
        """ Traverse the list of block and store each block's neighbours
        for speedier grading/count propagation """
        for axis in range(3):
            axis_pairs = block.get_axis_vertex_pairs(axis)

            for candidate in self.blocks:
                if candidate == block:
                    continue

                for pair in axis_pairs:
                    b_axis, _ = candidate.get_axis_from_pair(pair)
                    if b_axis is not None:
                        # the 'candidate' shares the same edge or face
                        block.neighbours.add(candidate)

    def copy_grading(self, block:Block, axis:List) -> bool:
        """ Finds a block that shares an edge with given block
        and copies its grading along that axis """
        # there are 4 pairs of vertices on specified axis:
        match_pairs = block.get_axis_vertex_pairs(axis)

        # first, find a block in mesh that shares one of the
        # edges in match_pairs:
        for n in block.neighbours:
            for p in match_pairs:
                b_axis, direction = n.get_axis_from_pair(p)
                if b_axis is not None:
                    # b.get_axis_from_pair() returns axis index in
                    # the block we want to copy from;
                    if n.grading[b_axis].is_defined:
                        # this block's count/grading is defined on this axis
                        # so we can (must) copy it
                        block.grading[axis] = n.grading[b_axis].copy(invert=not direction)
                        
                        return True
        return False

    def collect_vertices(self) -> NoReturn:
        """ Collects all vertices from all blocks,
        checks for duplicates and gives them indexes """
        for block in self.blocks:
            for i, block_vertex in enumerate(block.vertices):
                found_vertex = self.find_vertex(block_vertex)

                if found_vertex is None:
                    block.vertices[i].mesh_index = len(self.vertices)
                    self.vertices.append(block_vertex)
                else:
                    block.vertices[i] = found_vertex

        # special treatment for merged patches:
        # duplicate all points that define slave patches
        duplicated_points = {} # { original_index:new_vertex }
        slave_patches = [mp[1] for mp in self.merged_patches]

        for patch in slave_patches:
            for block in self.blocks: 
                if patch in block.patches:
                    for face in block.get_faces(patch, internal=True):
                        for i in range(4):
                            i_vertex = face[i]

                            vertex = block.vertices[i_vertex]

                            if vertex.mesh_index not in duplicated_points:
                                new_vertex = Vertex(vertex.point)
                                new_vertex.mesh_index = len(self.vertices)
                                self.vertices.append(new_vertex)

                                block.vertices[i_vertex] = new_vertex

                                duplicated_points[vertex.mesh_index] = new_vertex
                            else:
                                block.vertices[i_vertex] = duplicated_points[vertex.mesh_index]

    def collect_edges(self) -> NoReturn:
        """ Collects all edges from all blocks;
        checks for duplicates (same vertex pairs) and
        validity (no lines or straight-line arcs);
        Removes edges that don't pass those tests """
        for block in self.blocks:
            legit_edges = []

            for i, block_edge in enumerate(block.edges):
                # block.vertices by now contain index to mesh.vertices;
                # edge vertex therefore refers to actual mesh vertex
                v_1 = block.vertices[block_edge.block_index_1]
                v_2 = block.vertices[block_edge.block_index_2]

                block.edges[i].vertex_1 = v_1
                block.edges[i].vertex_2 = v_2

                if block_edge.type == 'line':
                    continue

                if block_edge.is_valid:
                    if self.find_edge(v_1, v_2) is None:
                        legit_edges.append(block_edge)

            self.edges += legit_edges
            block.edges = legit_edges

    def collect_neighbours(self) -> NoReturn:
        """ Collects all neighbours from all blocks;
        when setting counts and gradings, each block will iterate over them
        only and not through all blocks to save on time """
        for block in self.blocks:
            self.assign_neighbours(block)

    def set_gradings(self) -> NoReturn:
        """ Sets cell counts and grading
        (maths and stuff is in Grading object) """
        for block in self.blocks:
            block.grade()

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
                    if not block.grading[axis].is_defined:
                        updated = self.copy_grading(block, axis) or updated

                if block.is_grading_defined:
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

    def project_faces(self) -> NoReturn:
        """ Fills the self.faces list """
        self.faces = []
        for b in self.blocks:
            # TODO: check for existing faces
            for f in b.faces:
                self.faces.append([
                    b.get_face(f[0]), # face, like (8 12 15 11) 
                    f[1] # the geometry to project to
                ])

    def prepare_data(self, debug:bool=False) -> NoReturn:
        """ Converts the data from Block objects to a mountain of spaghetti
        to be written to blockMeshDict """
        self.collect_vertices()

        if debug:
            self.to_vtk()

        self.collect_edges()
        self.collect_neighbours()
        self.set_gradings()
        self.project_faces()

        self.patches = self.get_patches()

    def merge_patches(self, master:str, slave:str) -> NoReturn:
        """ Adds a master/slave pair of patches to this mesh """
        self.merged_patches.append([master, slave])

    def set_default_patch(self, patch_name:str, patch_type:str) -> NoReturn:
        """ Sets the defaultPatch """
        assert patch_type in ('patch', 'wall', 'empty', 'wedge')

        self.default_patch = {
            'name': patch_name,
            'type': patch_type
        }

    def add_geometry(self, g:dict) -> NoReturn:
        """ Adds an entry to a list of geometry objects for face/edge projections.
        blockMesh supports triangulated surfaces as well as all searchable surfaces
        that snappyHexMesh supports:
        https://www.openfoam.com/documentation/guides/latest/doc/guide-meshing-snappyhexmesh-geometry.html

        See the documentation for each type of surface you want to use and provice
        a python dictionary of surfaces (key=name) and a value that is a list of
        required attributes. For example:

        geometry = {
            'terrain': [
                'type triSurfaceMesh',
                'name terrain',
                'file "terrain.stl"',
            ],
            'tower': [
                'type         searchableCone',
                'point1       (0 0 0)',
                'radius1      1.5',
                'innerRadius1 0',
                'point2       (0 0 10)',
                'radius2      1.0',
                'innerRadius2 0',
            ]
        }

        Use f-strings for easier formatting of numerical attributes
        https://docs.python.org/3/reference/lexical_analysis.html#f-strings
        """
        # TODO: TEST
        self.geometry = {**self.geometry, **g}

    def write(self, output_path:str, template_path:str=None, debug:bool=True) -> NoReturn:
        """ Writes a blockMeshDict file to output_path.
        If template_path is not given, a generic one is used (provided with classy_blocks.
        
        The debug flag switches writing of a debug.ctk file with blocking - it can be opened 
        with ParaView, then colored by block_ids for easier debugging when blockMesh crashes. """
        # if template path is not given, find the default relative to this file
        if template_path is None:
            classy_dir = os.path.dirname(__file__)
            template_path = os.path.join(classy_dir, '..', 'util', 'blockMeshDict.template')

        self.prepare_data(debug)

        context = {
            'vertices': self.vertices,
            'edges': self.edges,
            'blocks': self.blocks,
            'patches': self.patches,
            'faces': self.faces,
            'default_patch': self.default_patch,
            'merged_patches': self.merged_patches,
            'geometry': self.geometry,
        }

        tools.template_to_dict(template_path, output_path, context)

    def to_vtk(self) -> NoReturn:
        """ Creates a VTK file with each mesh.block represented as a hexahedron,
        useful for debugging when Mesh.write() succeeds but blockMesh fails.
        Can only be called after Mesh.write() has been successfully finished! """
        context = {
            'points': [v.point for v in self.vertices],
            'cells': [[v.mesh_index for v in b.vertices] for b in self.blocks]
        }

        tools.template_to_dict(
            os.path.join(os.path.dirname(__file__), '..', 'util', 'vtk.template'),
            self.output_path, context)
