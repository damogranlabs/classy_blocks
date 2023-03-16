from typing import Dict, List, Optional

from classy_blocks.types import OrientType
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.items.side import Side
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.patch import Patch

class PatchList:
    """Handling of the patches ('boundary') part of blockMeshDict"""
    def __init__(self):
        self.patches:Dict[str, Patch] = {} # TODO: OrderedDict for consistent testing?
        self.default:Optional[Dict[str, str]] = None

    def add(self, vertices:List[Vertex], operation:Operation) -> None:
        """Create Patches from operation's patch_names"""
        for orient in operation.patch_names:
            self.add_side(operation.patch_names[orient], orient, vertices)

    def add_side(self, patch_name:str, orient:OrientType, vertices:List[Vertex]) -> None:
        """Adds a quad to an existing patch or creates a new one"""
        if patch_name not in self.patches:
            self.patches[patch_name] = Patch(patch_name)

        self.patches[patch_name].add_side(Side(orient, vertices))

    def set_default(self, name:str, kind:str) -> None:
        """Creates the default Patch"""
        self.default = {'name': name, 'kind': kind }

    def modify(self, name:str, kind:str, settings:Optional[List[str]]=None) -> None:
        """Changes patch's properties"""
        self.patches[name].kind = kind

        if settings is not None:
            self.patches[name].settings = settings

    @property
    def description(self) -> str:
        """Outputs a 'boundary' and 'faces' dict to be inserted directly into blockMeshDict"""
        out = "boundary\n(\n"

        for _, patch in self.patches.items():
            out += patch.description

        out += ");\n\n"

        if self.default is not None:
            out += "defaultPatch\n{\n"
            out += f"\tname {self.default['name']};\n"
            out += f"\ttype {self.default['kind']};\n"
            out += "}\n\n"

        return out
