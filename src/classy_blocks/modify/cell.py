from typing import Dict, List, Optional, Set

from classy_blocks.items.block import Block
from classy_blocks.items.vertex import Vertex
from classy_blocks.types import OrientType
from classy_blocks.util import constants


class NoCommonSidesError(Exception):
    """Raised when two cells don't share a side"""


class Cell:
    """A block, treated as a single cell;
    its quality metrics can then be transcribed directly
    from checkMesh."""

    def __init__(self, block: Block):
        self.block = block
        self.neighbours: Dict[OrientType, Optional[Cell]] = {
            "bottom": None,
            "top": None,
            "left": None,
            "right": None,
            "front": None,
            "back": None,
        }

        self.vertex_indexes = [self.block.vertices[i].index for i in range(8)]

    def get_common_vertices(self, candidate: "Cell") -> Set[int]:
        """Returns indexes of common vertices between this and provided cell"""
        this_indexes = set(self.vertex_indexes)
        cnd_indexes = {vertex.index for vertex in candidate.block.vertices}

        return this_indexes.intersection(cnd_indexes)

    def get_corner(self, vertex_index: int) -> int:
        """Converts vertex index to local index
        (position of this vertex in the list)"""
        return self.vertex_indexes.index(vertex_index)

    def get_common_side(self, candidate: "Cell") -> OrientType:
        """Returns orient of this cell that is shared with candidate"""
        common_vertices = self.get_common_vertices(candidate)

        if len(common_vertices) != 4:
            raise NoCommonSidesError

        corners = {self.get_corner(i) for i in common_vertices}

        for orient, indexes in constants.FACE_MAP.items():
            if set(indexes) == corners:
                return orient

        raise NoCommonSidesError

    def add_neighbour(self, candidate: "Cell") -> bool:
        """Adds the provided block to appropriate
        location in self.neighbours and returns True if
        this and provided block share a face;
        does nothing and returns False otherwise"""

        if candidate == self:
            return False

        try:
            orient = self.get_common_side(candidate)
            self.neighbours[orient] = candidate
            return True
        except NoCommonSidesError:
            return False

    @property
    def vertices(self) -> List[Vertex]:
        return self.block.vertices
