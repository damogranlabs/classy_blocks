import dataclasses
from typing import List


from classy_blocks.items.side import Side
from classy_blocks.items.vertex import Vertex
from classy_blocks.util import constants

@dataclasses.dataclass
class Face:
    """Block 'side' is defined by orient but a Face is defined by actual
    Vertex indexes; it also contains patch/project and formatting information"""
    vertices:List[Vertex]
    side:Side

    def __post_init__(self):
        assert len(self.vertices) == 4, "A Face must contain exactly 4 Vertices"

    @property
    def description(self) -> str:
        """Outputs a string that represents a block face in blockMeshDict"""
        indexes = ' '.join([str(v.index) for v in self.vertices])

        return '(' + indexes + ')'

    @classmethod
    def from_side(cls, side:Side, vertices:List[Vertex]) -> 'Face':
        """Create a face from given Side object"""
        vertices = [vertices[i] for i in constants.FACE_MAP[side.orient]]

        return cls(vertices, side)
