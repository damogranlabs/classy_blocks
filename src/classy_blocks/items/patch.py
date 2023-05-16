import warnings
from typing import List

from classy_blocks.items.side import Side
from classy_blocks.util.tools import indent


class Patch:
    """Definition of a patch, including type, belonging faces and other settings"""

    def __init__(self, name: str):
        self.name = name

        self.sides: List[Side] = []

        self.kind = "patch"  # 'type'
        self.settings: List[str] = []

    def add_side(self, side: Side) -> None:
        """Adds a side to the list if it doesn't exist yet"""
        for existing in self.sides:
            if existing == side:
                warnings.warn(f"Side {side.description} has already been assigned to {self.name}", stacklevel=2)
                return

        self.sides.append(side)

    @property
    def description(self) -> str:
        """patch definition for blockMeshDict"""
        # inlet
        # {
        #     type patch;
        #     faces
        #     (
        #         (0 1 2 3)
        #     );
        # }
        out = indent(self.name, 1)
        out += indent("{", 1)
        out += indent(f"type {self.kind};", 2)

        for option in self.settings:
            out += indent(f"{option};", 2)

        out += indent("faces", 2)
        out += indent("(", 2)

        for quad in self.sides:
            out += indent(f"{quad.description}", 3)

        out += indent(");", 2)
        out += indent("}", 1)

        return out
