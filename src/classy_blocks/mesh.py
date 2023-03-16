"""The Mesh object ties everything together and writes the blockMeshDict in the end."""
from typing import  Optional, List, Dict

from classy_blocks.items.vertex import Vertex
from classy_blocks.items.block import Block
from classy_blocks.lists.block_list import BlockList
from classy_blocks.lists.vertex_list import VertexList
from classy_blocks.lists.edge_list import EdgeList
from classy_blocks.lists.patch_list import PatchList
from classy_blocks.lists.face_list import FaceList
from classy_blocks.lists.geometry_list import GeometryList

from classy_blocks.construct.operations.operation import Operation

from classy_blocks.base.additive import AdditiveBase

from classy_blocks.util import constants
from classy_blocks.util.tools import write_vtk

class Mesh:
    """contains blocks, edges and all necessary methods for assembling blockMeshDict"""
    def __init__(self):
        self.vertex_list = VertexList()
        self.edge_list = EdgeList()
        self.block_list = BlockList()
        self.patch_list = PatchList()
        self.face_list = FaceList()
        self.geometry_list = GeometryList()

        self.settings = {
            # TODO: test output
            'prescale': None,
            'scale': 1,
            'transform': None,
            'mergeType': None, # use 'points' to fall back to the older point-based block merging 
            'checkFaceCorrespondence': None, # true by default, turn off if blockMesh fails (3-sided pyramids etc.)
            'verbose': None,
        }

    def add(self, entity:AdditiveBase) -> None:
        """Add a classy_blocks entity to the mesh;
        can be a plain Block, created from points, Operation, Shape or Object."""
        for operation in entity.operations:
            self.add_operation(operation)

    def add_operation(self, operation:Operation) -> None:
        """Takes an operation, converts it to Block and adds that to the mesh"""
        vertices = self._add_vertices(operation)

        block = Block(len(self.block_list.blocks), vertices)
        self._add_edges(operation, vertices, block)
        self._chop_block(operation, block)
        self.block_list.add(block)

        self._add_patches(vertices, operation)

        self._project_faces(vertices, operation)

        # TODO: TEST
        #if hasattr(item, "geometry"):
        #    raise NotImplementedError
        #   # self.add_geometry(item.geometry)

    def _add_vertices(self, operation:Operation) -> List[Vertex]:
        """Creates/finds vertices from operation's points and returns them"""
        return [self.vertex_list.add(p) for p in operation.bottom_face.points] + \
            [self.vertex_list.add(p) for p in operation.top_face.points]

    def _add_edges(self, operation:Operation, vertices:List[Vertex], block:Block) -> None:
        """Creates/finds edges from operation and returns them"""
        edges = self.edge_list.add(vertices, operation.bottom_face.edges, 'bottom') + \
            self.edge_list.add(vertices, operation.top_face.edges, 'top') + \
            self.edge_list.add(vertices, operation.side_edges, 'side')
        
        for edge in edges:
            block.add_edge(*edge)

    def _chop_block(self, operation:Operation, block:Block) -> None:
        """Chops the block as declared in Operation"""
        for axis in (0, 1, 2):
            for chop in operation.chops[axis]:
                block.chop(axis, chop)

    def _add_patches(self, vertices:List[Vertex], operation:Operation) -> None:
        """Creates patches and projects faces"""
        self.patch_list.add(vertices, operation)

    def _project_faces(self, vertices:List[Vertex], operation:Operation) -> None:
        """Collects projected faces from operation"""
        self.face_list.add(vertices, operation)

    # def merge_patches(self, master:str, slave:str) -> None:
    #     """Merges two non-conforming named patches using face merging;
    #     https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.3-mesh-generation-with-the-blockmesh-utility#x13-470004.3.2
    #     (breaks the 100% hex-mesh rule)"""
    #     self.patches['merged'].append([master, slave])

    def set_default_patch(self, name:str, kind:str) -> None:
        """Adds the 'defaultPatch' entry to the mesh; any non-specified block boundaries
        will be assigned this patch"""
        self.patch_list.set_default(name, kind)

    def modify_patch(self, name:str, kind:str, settings:Optional[List[str]]=None) -> None:
        """Fetches a patch named 'patch' and modifies its type and optionally
        other settings. They are passed on to blockMeshDict as a list of strings
        as-is, with no additional brain power used"""
        self.patch_list.modify(name, kind, settings)

    def add_geometry(self, geometry:dict) -> None:
        """Adds named entry in the 'geometry' section of blockMeshDict;
        'g' is in the form of dictionary {'geometry_name': [list of properties]};
        properties are as specified by searchable* class in documentation.
        See examples/advanced/project for an example."""
        self.geometry_list.add(geometry)

    def write(self, output_path:str, debug_path:Optional[str]=None) -> None:
        """Writes a blockMeshDict to specified location. If debug_path is specified,
        a VTK file is created first where each block is a single cell, to see simplified
        blocking in case blockMesh fails with an unfriendly error message."""
        if debug_path is not None:
           write_vtk(debug_path, self.vertex_list.vertices, self.block_list.blocks)

        self.block_list.propagate_gradings()

        with open(output_path, 'w', encoding='utf-8') as output:
            output.write(constants.MESH_HEADER)

            for key, value in self.settings.items():
                if value is not None:
                    output.write(f"{key} {value};\n")
            output.write('\n')

            output.write(self.geometry_list.description)

            output.write(self.vertex_list.description)
            output.write(self.block_list.description)
            output.write(self.edge_list.description)
            output.write(self.face_list.description)
            output.write(self.patch_list.description)

            # merged patches: output manually
            # if len(self.patches['merged']) > 0:
            #     f.write("mergePatchPairs\n(\n")
            #     for pair in self.patches['merged']:
            #         f.write(f"\t({pair[0]} {pair[1]})\n")
                
            #     f.write(");\n\n")

            output.write(constants.MESH_FOOTER)

