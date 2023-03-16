import warnings

from typing import List, Dict

from classy_blocks.items.side import Side

class Patch:
    """Definition of a patch, including type, belonging faces and other settings"""
    def __init__(self, name:str):
        self.name = name

        self.sides:List[Side] = []

        self.kind = 'patch' # 'type'
        self.settings:List[str] = []

    def add_side(self, side:Side) -> None:
        """Adds a side to the list if it doesn't exist yet"""
        for existing in self.sides:
            if existing == side:
                warnings.warn(f"Side {side.description} has already been assigned to {self.name}")
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
        # TODO: do something with all that \t\n\n};\t\t chaos
        out = "\t" + self.name + "\n\t{\n"
        out += f"\t\ttype {self.kind};\n"
        out += "\t\tfaces\n\t\t(\n"

        for quad in self.sides:
            out += f"\t\t\t{quad.description}\n"

        for option in self.settings:
            out += f"\n\t\t{option};"

        out += "\t\t);"
        out += "\n\t}\n"

        return out
