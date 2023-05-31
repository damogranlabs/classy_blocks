from typing import Optional


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

    def __init__(self, msg: str, details: Optional[str] = None, *args) -> None:
        self.msg = msg
        self.details = details

        info = self.msg
        if self.details:
            info += f"\n\t{self.details}"

        super().__init__(info, *args)


class PointCreationError(ShapeCreationError):
    pass


class AnnulusCreationError(ShapeCreationError):
    pass


class EdgeCreationError(ShapeCreationError):
    pass


class SideCreationError(ShapeCreationError):
    pass


class FaceCreationError(ShapeCreationError):
    pass


class CylinderCreationError(ShapeCreationError):
    pass


class ElbowCreationError(ShapeCreationError):
    pass


class FrustumCreationError(ShapeCreationError):
    pass


class ExtrudedRingCreationError(ShapeCreationError):
    pass
