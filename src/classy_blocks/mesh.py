"""The Mesh object ties everything together and writes the blockMeshDict in the end."""
from typing import Union, Optional

from classy_blocks.data.block import BlockData

from classy_blocks.items.vertex import Vertex
from classy_blocks.items.block import Block

from classy_blocks.lists.block_list import BlockList
from classy_blocks.lists.vertex_list import VertexList
from classy_blocks.lists.edge_list import EdgeList

from classy_blocks.construct.operations import Operation
from classy_blocks.construct.shapes import Shape

from classy_blocks.util import constants

class Mesh:
    """contains blocks, edges and all necessary methods for assembling blockMeshDict"""
    def __init__(self):
        self.block_list = BlockList()

        self.vertex_list = VertexList()
        self.edge_list = EdgeList()

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

            #if debug_path is not None:
            #    tools.write_vtk(debug_path, self.vertices, self.blocks)

            #self.boundary.collect(self.blocks)
            #self.faces.collect(self.blocks)
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
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(constants.MESH_HEADER)

            for key, value in self.settings.items():
                if value is not None:
                    f.write(f"{key} {value};\n")
            f.write('\n')
            
            #f.write(self.geometry.output())

            f.write(self.vertex_list.description)
            f.write(self.block_list.description)
            f.write(self.edge_list.description)
            #f.write("edges\n();\n\n")

            #f.write(self.boundary.output())
            f.write("boundary\n();\n\n")
            #f.write(self.faces.output())
            f.write("faces\n();\n\n")

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

