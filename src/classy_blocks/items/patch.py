import dataclasses

from typing import List

from classy_blocks.items.face import Face

@dataclasses.dataclass
class Patch:
    """Definition of a patch, including type, 
    belonging faces and other settings"""
    name:str
    faces:List[Face]
    type:str
    settings:dict = dataclasses.field(default_factory=dict)

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
        out += f"\t\ttype {self.type};\n"
        out += "\t\tfaces\n\t\t(\n"

        for face in self.faces:
            out += f"\t\t\t{face.description}\n"

        for key, value in self.settings.items():
            out += f"\n\t\t{key} {value};"

        out += "\t\t);"
        out += "\n\t}\n"

        return out