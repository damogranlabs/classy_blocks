from collections import OrderedDict
from typing import Dict, List, Optional, Set

from classy_blocks.construct.operations.operation import Operation
from classy_blocks.items.patch import Patch
from classy_blocks.items.side import Side
from classy_blocks.items.vertex import Vertex
from classy_blocks.types import OrientType


class PatchList:
    """Handling of the patches ('boundary') part of blockMeshDict"""

    def __init__(self) -> None:
        self.patches: OrderedDict[str, Patch] = OrderedDict()
        self.default: Dict[str, str] = {}
        self.merged: List[List[str]] = []  # data for the mergePatchPairs entry

    def add(self, vertices: List[Vertex], operation: Operation) -> None:
        """Create Patches from operation's patch_names"""
        for orient, name in operation.patch_names.items():
            self.add_side(name, orient, vertices)

    def get(self, name: str) -> Patch:
        """Fetches an existing Patch or creates a new one"""
        if name not in self.patches:
            self.patches[name] = Patch(name)

        return self.patches[name]

    def add_side(self, patch_name: str, orient: OrientType, vertices: List[Vertex]) -> None:
        """Adds a quad to an existing patch or creates a new one"""
        self.get(patch_name).add_side(Side(orient, vertices))

    def set_default(self, name: str, kind: str) -> None:
        """Creates the default Patch"""
        self.default = {"name": name, "kind": kind}

    def modify(self, name: str, kind: str, settings: Optional[List[str]] = None) -> None:
        """Changes patch's properties"""
        patch = self.get(name)
        patch.kind = kind

        if settings is not None:
            patch.settings = settings

    def merge(self, master: str, slave: str) -> None:
        """Adds an entry in mergePatchPairs list in blockMeshDict"""
        self.merged.append([master, slave])

    def clear(self) -> None:
        """Removes collected patches but leaves settings intact"""
        self.patches.clear()

    @property
    def description(self) -> str:
        """Outputs a 'boundary' and 'faces' dict to be inserted directly into blockMeshDict"""
        out = "boundary\n(\n"

        for _, patch in self.patches.items():
            out += patch.description

        out += ");\n\n"

        if self.default:
            out += "defaultPatch\n{\n"
            out += f"\tname {self.default['name']};\n"
            out += f"\ttype {self.default['kind']};\n"
            out += "}\n\n"

        # merged patches
        out += "mergePatchPairs\n(\n"
        for pair in self.merged:
            out += f"\t({pair[0]} {pair[1]})\n"
        out += ");\n\n"

        return out

    @property
    def master_patches(self) -> Set[str]:
        """A list of master patch names"""
        return {patches[0] for patches in self.merged}

    @property
    def slave_patches(self) -> Set[str]:
        """A list of master patch names"""
        return {patches[1] for patches in self.merged}

    def is_slave(self, name: str) -> bool:
        """Returns True if a patch with given name is
        listed as a slave in merged patches"""
        return name in self.slave_patches
