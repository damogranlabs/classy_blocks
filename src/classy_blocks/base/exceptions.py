from classy_blocks.types import PointType


class VertexNotFoundError(Exception):
    """Raised when a vertex at a given point in space doesn't exist yet"""


class EdgeNotFoundError(Exception):
    """Raised when an edge between a given pair of vertices doesn't exist yet"""


class CornerPairError(Exception):
    """Raised when given pair of corners is not valid (for example, edge between 0 and 2)"""


class UndefinedGradingsError(Exception):
    """Raised when the user hasn't supplied enough grading data to
    define all blocks in the mesh"""


class ShapeCreationError(Exception):
    """Base class for shape creation errors (invalid parameters/types to
    shape constructors)"""


class PointCreationError(ShapeCreationError):
    def __init__(self, msg: str, position: PointType):
        self.position = position

        super().__init__(f"{msg}\n\tPosition: {position}")
