from typing import List

from classy_blocks.process.items.vertex import Vertex
from classy_blocks.data.block import BlockData


class Boundary:
    """Handling of the 'boundary' part of blockMeshDict"""
    def __init__(self):
        # A collection {'patch_name': [lists of [list of 4 Vertex objects]]}
        self.patches:dict = {}

    def collect(self, blocks:List[BlockData]) -> None:
        """Collects all faces for a patch name from all blocks;
        Block contains patches according to the example in __init__()"""

        # collect all different patch names
        patch_names = set()
        for block in blocks:
            patch_names = patch_names.union(set(block.patches.keys()))

        # create a dict with all patches
        self.patches = {name: [] for name in patch_names}

        # gather all faces of all blocks
        for block in blocks:
            for patch_name in patch_names:
                orients = block.get_patch_sides(patch_name)
                self.patches[patch_name] += [block.get_side_vertices(o) for o in orients]

    def output(self) -> str:
        """Outputs a 'boundary' dict to be inserted directly into blockMeshDict"""
        bnd = "boundary\n(\n"

        for name, faces in self.patches.items():
            bnd += f"\t{name}\n"
            bnd += "\t{\n"
            bnd += "\t\ttype patch;\n" # TODO: different patch types & other properties
            bnd += "\t\tfaces\n"
            bnd += "\t\t(\n"

            for face in faces:
                bnd += f"\t\t\t{self.format_face(face)}\n"

            bnd += "\t\t);\n"
            bnd += "\t}\n"

        bnd += ");\n\n"

        return bnd

    @staticmethod
    def format_face(vertices:list[Vertex]) -> str:
        """Outputs a string that represents a block face in blockMeshDict"""
        assert len(vertices) == 4

        indexes = [str(v.mesh_index) for v in vertices]

        return f"({' '.join(indexes)})"
