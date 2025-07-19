from collections import OrderedDict
from typing import Optional

from classy_blocks.base.exceptions import PatchNotFoundError
from classy_blocks.cbtyping import OrientType
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.items.patch import Patch
from classy_blocks.items.side import Side
from classy_blocks.items.vertex import Vertex


class PatchList:
    """Handling of the patches ('boundary') part of blockMeshDict"""

    def __init__(self) -> None:
        self.patches: OrderedDict[str, Patch] = OrderedDict()
        self.default: dict[str, str] = {}
        self.merged: list[list[str]] = []  # data for the mergePatchPairs entry

    def add(self, vertices: list[Vertex], operation: Operation) -> None:
        """Create Patches from operation's patch_names"""
        for orient, name in operation.patch_names.items():
            self.add_side(name, orient, vertices)

    def get(self, name: str) -> Patch:
        """Fetches an existing Patch or creates a new one"""
        if name not in self.patches:
            self.patches[name] = Patch(name)

        return self.patches[name]

    def find(self, vertices: set[Vertex]) -> Patch:
        # TODO: use FaceRegistry
        for patch in self.patches.values():
            for side in patch.sides:
                if set(side.vertices) == vertices:
                    return patch

        raise PatchNotFoundError

    def add_side(self, patch_name: str, orient: OrientType, vertices: list[Vertex]) -> None:
        """Adds a quad to an existing patch or creates a new one"""
        self.get(patch_name).add_side(Side(orient, vertices))

    def modify(self, name: str, kind: str, settings: Optional[list[str]] = None) -> None:
        """Changes patch's properties"""
        patch = self.get(name)
        patch.kind = kind

        if settings is not None:
            patch.settings = settings

    def merge(self, master: str, slave: str) -> None:
        """Adds an entry in mergePatchPairs list in blockMeshDict"""
        self.merged.append([master, slave])
