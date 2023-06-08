from typing import List

from classy_blocks.items.vertex import Vertex
from classy_blocks.types import OrientType
from classy_blocks.util import constants


class Side:
    """A Block has 6 'sides', defined by vertices but addressed by OrientType;
    new faces can be created from those and patches are assigned to this"""

    def __init__(self, orient: OrientType, vertices: List[Vertex]):
        assert len(vertices) == 8, "Pass all 8 of block's vertices"

        corners = constants.FACE_MAP[orient]
        self.vertices = [vertices[i] for i in corners]

    @property
    def description(self) -> str:
        """Outputs a string that represents a block face in blockMeshDict"""
        indexes = " ".join([str(v.index) for v in self.vertices])

        return "(" + indexes + ")"

    def __eq__(self, other):
        return {v.index for v in self.vertices} == {v.index for v in other.vertices}
