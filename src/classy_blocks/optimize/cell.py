import abc
from typing import ClassVar, Optional

from classy_blocks.base.exceptions import NoCommonSidesError
from classy_blocks.cbtyping import IndexType, NPPointListType, OrientType
from classy_blocks.optimize.connection import CellConnection
from classy_blocks.util.constants import EDGE_PAIRS


class CellBase(abc.ABC):
    side_names: ClassVar[list[OrientType]]
    side_indexes: ClassVar[list[IndexType]]
    edge_pairs: ClassVar[list[tuple[int, int]]]

    def __init__(self, index: int, grid_points: NPPointListType, indexes: IndexType):
        self.index = index
        self.grid_points = grid_points
        self.indexes = indexes

        self.neighbours: dict[OrientType, Optional[CellBase]] = {name: None for name in self.side_names}
        self.connections = [CellConnection(set(pair), {indexes[pair[0]], indexes[pair[1]]}) for pair in self.edge_pairs]

    def get_common_indexes(self, candidate: "CellBase") -> set[int]:
        """Returns indexes of common vertices between this and provided cell"""
        this_indexes = set(self.indexes)
        cnd_indexes = set(candidate.indexes)

        return this_indexes.intersection(cnd_indexes)

    def get_corner(self, index: int) -> int:
        """Converts vertex index to local index
        (position of this vertex in the list)"""
        return self.indexes.index(index)

    def get_common_side(self, candidate: "CellBase") -> OrientType:
        """Returns orient of this cell that is shared with candidate"""
        common_vertices = self.get_common_indexes(candidate)

        if len(common_vertices) != len(self.side_indexes[0]):
            raise NoCommonSidesError

        corners = {self.get_corner(i) for i in common_vertices}

        for i, indexes in enumerate(self.side_indexes):
            if set(indexes) == corners:
                return self.side_names[i]

        raise NoCommonSidesError

    @property
    def boundary(self) -> set[int]:
        """Returns a list of indexes that define sides on boundary"""
        boundary = set()

        for i, side_name in enumerate(self.side_names):
            side_indexes = self.side_indexes[i]

            if self.neighbours[side_name] is None:
                boundary.update({self.indexes[si] for si in side_indexes})

        return boundary

    def __str__(self):
        return "-".join([str(index) for index in self.indexes])

    def __repr__(self):
        return str(self)


class QuadCell(CellBase):
    # Like constants.FACE_MAP but for quadrangle sides as line segments
    side_names: ClassVar = ["front", "right", "back", "left"]
    side_indexes: ClassVar = [[0, 1], [1, 2], [2, 3], [3, 0]]
    edge_pairs: ClassVar = [(0, 1), (1, 2), (2, 3), (3, 0)]


class HexCell(CellBase):
    """A block, treated as a single cell;
    its quality metrics can then be transcribed directly
    from checkMesh."""

    # FACE_MAP, ordered and modified so that all faces point towards cell center;
    # provided their points are visited in an anti-clockwise manner
    # names and indexes must correspond (both must be in the same order)
    side_names: ClassVar = ["bottom", "top", "left", "right", "front", "back"]
    side_indexes: ClassVar = [[0, 1, 2, 3], [7, 6, 5, 4], [4, 0, 3, 7], [6, 2, 1, 5], [0, 4, 5, 1], [7, 3, 2, 6]]
    edge_pairs: ClassVar = EDGE_PAIRS
