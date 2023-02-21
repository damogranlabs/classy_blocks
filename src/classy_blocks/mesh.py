"""The Mesh object ties everything together and writes the blockMeshDict in the end."""
from typing import Union, Optional

from classy_blocks.data.block_data import BlockData

from classy_blocks.items.block import Block

from classy_blocks.lists.block_list import BlockList
from classy_blocks.lists.vertex_list import VertexList
from classy_blocks.lists.edge_list import EdgeList
from classy_blocks.lists.boundary import Boundary

from classy_blocks.construct.operations import Operation
from classy_blocks.construct.shapes import Shape

from classy_blocks.util import constants
from classy_blocks.util.tools import write_vtk

class Mesh:
    """contains blocks, edges and all necessary methods for assembling blockMeshDict"""
    def __init__(self):
        self.vertex_list = VertexList()
        self.block_list = BlockList()
        self.edge_list = EdgeList()
        self.boundary = Boundary()

        self.settings = {
            # TODO: test output
            'prescale': None,
            'scale': 1,
            'transform': None,
            'mergeType': None, # use 'points' to fall back to the older point-based block merging 
            'checkFaceCorrespondence': None, # true by default, turn off if blockMesh fails (3-sided pyramids etc.)
            'verbose': None,
        }

        self.patches = {
            'default': None,
            'merged': [],
        }

    def add(self, item:BlockData) -> None:
        """Add a classy_blocks entity to the mesh;
        can be a plain Block, created from points, Operation, Shape or Object."""
        # add blocks to block list
        for data in item.data:
            # generate Vertices from all block's points or find existing ones
            vertices = self.vertex_list.add(data.points)
            # generate new edges or find existing ones
            edges = self.edge_list.add(data, vertices)

            # generate a Block from collected/created objects
            block = Block(data, len(self.block_list.blocks), vertices, edges)
            self.block_list.add(block)

            # generate patches from block's faces
            #self.boundary.add(block)

            # TODO: TEST
            #if hasattr(item, "geometry"):
            #    raise NotImplementedError
            #   # self.add_geometry(item.geometry)

    # def merge_patches(self, master:str, slave:str) -> None:
    #     """Merges two non-conforming named patches using face merging;
    #     https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.3-mesh-generation-with-the-blockmesh-utility#x13-470004.3.2
    #     (breaks the 100% hex-mesh rule)"""
    #     self.patches['merged'].append([master, slave])

    # def set_default_patch(self, name:str, ptype:str) -> None:
    #     """Adds the 'defaultPatch' entry to the mesh; any non-specified block boundaries
    #     will be assigned this patch"""
    #     assert ptype in ("patch", "wall", "empty", "wedge")

    #     self.patches['default'] = {"name": name, "type": ptype}

    # def add_geometry(self, geometry:dict) -> None:
    #     """Adds named entry in the 'geometry' section of blockMeshDict;
    #     'g' is in the form of dictionary {'geometry_name': [list of properties]};
    #     properties are as specified by searchable* class in documentation.
    #     See examples/advanced/project for an example."""
    #     self.geometry.add(geometry)

    def write(self, output_path:str, debug_path:Optional[str]=None) -> None:
        """Writes a blockMeshDict to specified location. If debug_path is specified,
        a VTK file is created first where each block is a single cell, to see simplified
        blocking in case blockMesh fails with an unfriendly error message."""
        if debug_path is not None:
           write_vtk(debug_path, self.vertex_list.vertices, self.block_list.blocks)

        with open(output_path, 'w', encoding='utf-8') as output:
            output.write(constants.MESH_HEADER)

            for key, value in self.settings.items():
                if value is not None:
                    output.write(f"{key} {value};\n")
            output.write('\n')

            #f.write(self.geometry.output())

            output.write(self.vertex_list.description)
            output.write(self.block_list.description)
            output.write(self.edge_list.description)

            output.write(self.boundary.description)

            # patches: output manually
            # if len(self.patches['merged']) > 0:
            #     f.write("mergePatchPairs\n(\n")
            #     for pair in self.patches['merged']:
            #         f.write(f"\t({pair[0]} {pair[1]})\n")
                
            #     f.write(");\n\n")

            # if self.patches['default'] is not None:
            #     f.write("defaultPatch\n{\n")
            #     f.write(f"\tname {self.patches['default']['name']};\n")
            #     f.write(f"\ttype {self.patches['default']['type']};")
            #     f.write("\n}\n\n")

            output.write(constants.MESH_FOOTER)

