import warnings

from classy_blocks.items.side import Side


class Patch:
    """Definition of a patch, including type, belonging faces and other settings"""

    def __init__(self, name: str):
        self.name = name

        self.sides: set[Side] = set()

        self.kind = "patch"  # 'type'
        self.settings: list[str] = []

        self.slave = False

    def add_side(self, side: Side) -> None:
        """Adds a side to the list if it doesn't exist yet"""
        if side in self.sides:
            warnings.warn(f"Side {side} has already been assigned to {self.name}", stacklevel=2)
            return

        self.sides.add(side)
