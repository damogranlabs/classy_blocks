from typing import Dict, List, Optional

from classy_blocks.items.block import Block
from classy_blocks.items.vertex import Vertex
from classy_blocks.types import OrientType


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

    def add_neighbour(self, block: Block) -> bool:
        """Adds the provided block to appropriate
        location in self.neighbours and returns True if
        this and provided block share a face;
        does nothing and returns False otherwise"""
        return False

    @property
    def vertices(self) -> List[Vertex]:
        return self.block.vertices
