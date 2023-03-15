from typing import Dict, List

from classy_blocks.types import OrientType
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.items.side import Side
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.patch import Patch

# class FaceList:
#     """Handling of the 'faces' part of blockMeshDict (projected faces)"""
#     def __init__(self):
#         # a list of [[4 vertices], 'projected geometry']
#         self.faces:List = []

#     def collect(self, blocks:List[BlockData]) -> None:
#         """Gathers projected faces from blocks"""
#         for block in blocks:
#             # TODO: check for existing faces
#             for orient, side in block.sides.items():
#                 if side.project is not None:
#                     vertices = block.get_side_vertices(orient)
#                     self.faces.append([vertices, side.project])

#     def output(self) -> str:
#         """Formats the 'faces' list to be output into blockMeshDict"""
#         flist = "faces\n(\n"

#         for data in self.faces:
#             flist += f"\tproject {PatchList.format_face(data[0])} {data[1]}\n"

#         flist += ");\n\n"

#         return flist

class PatchList:
    """Handling of the 'boundary' part of blockMeshDict"""
    def __init__(self):
        self.patches:Dict[str, Patch] = {} # TODO: OrderedDict for consistent testing?

    def add(self, vertices:List[Vertex], operation:Operation) -> None:
        """Create Patches from operation's quads"""
        for orient in operation.patch_names:
            self.add_side(operation.patch_names[orient], orient, vertices)

    def add_side(self, patch_name:str, orient:OrientType, vertices:List[Vertex]) -> None:
        """Adds a quad to an existing patch or creates a new one"""
        if patch_name not in self.patches:
            self.patches[patch_name] = Patch(patch_name)

        self.patches[patch_name].add_side(Side(orient, vertices))

    @property
    def description(self) -> str:
        """Outputs a 'boundary' and 'faces' dict to be inserted directly into blockMeshDict"""
        out = "boundary\n(\n"

        for _, patch in self.patches.items():
            out += patch.description

        out += ");\n\n"

        return out

