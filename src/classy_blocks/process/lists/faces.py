from typing import List

from classy_blocks.data.block import BlockData
from classy_blocks.process.lists.boundary import Boundary

class FaceList:
    """Handling of the 'faces' part of blockMeshDict (projected faces)"""
    def __init__(self):
        # a list of [[4 vertices], 'projected geometry']
        self.faces:List = []

    def collect(self, blocks:List[BlockData]) -> None:
        """Gathers projected faces from blocks"""
        for block in blocks:
            # TODO: check for existing faces
            for orient, side in block.sides.items():
                if side.project is not None:
                    vertices = block.get_side_vertices(orient)
                    self.faces.append([vertices, side.project])

    def output(self) -> str:
        """Formats the 'faces' list to be output into blockMeshDict"""
        flist = "faces\n(\n"

        for data in self.faces:
            flist += f"\tproject {Boundary.format_face(data[0])} {data[1]}\n"

        flist += ");\n\n"

        return flist
