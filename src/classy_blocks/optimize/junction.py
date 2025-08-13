import dataclasses
from typing import Optional

import numpy as np

from classy_blocks.base.exceptions import ClampExistsError
from classy_blocks.cbtyping import NPPointListType, NPPointType
from classy_blocks.optimize.cell import CellBase, HexCell
from classy_blocks.optimize.clamps.clamp import ClampBase
from classy_blocks.optimize.links import LinkBase
from classy_blocks.optimize.quality import get_hex_quality, get_quad_quality


@dataclasses.dataclass
class LinkTie:
    leader: LinkBase
    follower_index: int


class Junction:
    """A class that collects Cells that
    share the same Vertex"""

    def __init__(self, points: NPPointListType, index: int):
        self.points = points
        self.index = index

        self.cells: set[CellBase] = set()

        self.neighbours: set[Junction] = set()

        self.clamp: Optional[ClampBase] = None
        self.links: list[LinkTie] = []

    @property
    def point(self) -> NPPointType:
        return self.points[self.index]

    def add_clamp(self, clamp: ClampBase) -> None:
        if self.clamp is not None:
            raise ClampExistsError(f"Clamp already defined for junction {self.index}")

        self.clamp = clamp

    def add_link(self, link: LinkBase, follower_index: int) -> None:
        self.links.append(LinkTie(link, follower_index))

    @property
    def is_boundary(self) -> bool:
        """Returns True if this junction lies on boundary"""
        for cell in self.cells:
            if self.index in cell.boundary:
                return True

        return False

    @property
    def quality(self) -> float:
        if isinstance(next(iter(self.cells)), HexCell):
            quality_function = get_hex_quality
        else:
            quality_function = get_quad_quality

        return sum(quality_function(self.points, np.array(cell.indexes, dtype=np.int32)) for cell in self.cells)
