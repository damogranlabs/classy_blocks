from typing import Dict, List

from classy_blocks.types import OrientType
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.items.side import Side
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.patch import Patch

class PatchList:
    """Handling of the patches ('boundary') part of blockMeshDict"""
    def __init__(self):
        self.patches:Dict[str, Patch] = {} # TODO: OrderedDict for consistent testing?

        # TODO:
        # self.default_patch = Patch('')
        #self.merged
        #'merged': [],
        #}

    def add(self, vertices:List[Vertex], operation:Operation) -> None:
        """Create Patches from operation's patch_names"""
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
