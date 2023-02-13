from typing import List

from classy_blocks.items.block import Block

from classy_blocks.items.patch import Patch

class Boundary:
    """Handling of the 'boundary' part of blockMeshDict"""
    def __init__(self):
        # A collection {'patch_name': [list of Faces]}
        self.patches:List[Patch] = []

    def find(self, name:str) -> Patch:
        """Find a patch in the list by name"""
        for patch in self.patches:
            if patch.name == name:
                return patch

        raise RuntimeError(f"Patch {name} not found")

    def add(self, block:Block) -> None:
        """Add patches from block to list"""
        for face in block.faces.values():
            if face.side.patch_name is not None:
                # this side specifies a patch;
                # find an existing and add this face or create a new one
                try:
                    patch = self.find(face.side.patch_name)
                    patch.faces.append(face)
                except RuntimeError:
                    # create a new patch and add a face to it
                    patch = Patch(face.side.patch_name, [face], face.side.patch_type)
                    self.patches.append(patch)

    @property
    def description(self) -> str:
        """Outputs a 'boundary' and 'faces' dict to be inserted directly into blockMeshDict"""
        out = "boundary\n(\n"

        for patch in self.patches:
            out += patch.description

        out += ");\n\n"

        out += "faces\n(\n"

        for patch in self.patches:
            for face in patch.faces:
                if face.side.project_to is not None:
                    out += f"\tproject {face.description}\n"
        out += ");\n\n"
                    
        return out

