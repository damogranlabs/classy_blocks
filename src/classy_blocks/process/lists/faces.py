from typing import List

from classy_blocks.define.block import Block
from classy_blocks.process.lists.boundary import Boundary

class FaceList:
    """Handling of the 'faces' part of blockMeshDict (projected faces)"""
    def output(self, blocks:List[Block]) -> str:
        """Formats the 'faces' list to be output into blockMeshDict"""
        flist = "faces\n(\n"

        for block in blocks:
           # TODO: check for existing faces
           for orient, side in block.sides.items():
                if side.project is not None:
                    face = block.get_side_vertices(orient)

                    flist += f"\tproject {Boundary.format_face(face)} {side.project}\n"
        
        flist += ");\n\n"

        return flist