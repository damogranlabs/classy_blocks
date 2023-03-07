from typing import Optional, List, Tuple

from classy_blocks.types import OrientType
from classy_blocks.items.vertex import Vertex
from classy_blocks.util import constants

class Side:
    """A Block has 6 'sides', defined by vertices but addressed by OrientType;
    new faces can be created from those and patches are assigned to this"""
    def __init__(self, vertices:List[Vertex], orient:OrientType):
        assert len(vertices) == 8, "Pass all 8 of block's vertices"

        self.orient:OrientType = orient
        self.vertices = [vertices[i] for i in self.corners]

        # to which patch this block side belongs (if any)
        self.patch_name:Optional[str] = None
        self.patch_type:str = 'patch'

        # project to a named searchable surface?
        self.project_to:Optional[List[str]] = None

    @property
    def corners(self) -> Tuple[int, int, int, int]:
        """Returns indexes of block-local indexes that define this side"""
        return constants.FACE_MAP[self.orient]

    @property
    def description(self) -> str:
        """Outputs a string that represents a block face in blockMeshDict"""
        indexes = ' '.join([str(v.index) for v in self.vertices])

        return '(' + indexes + ')'
