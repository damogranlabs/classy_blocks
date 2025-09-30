from typing import Optional


### Construction
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


class ArrayCreationError(ShapeCreationError):
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


class GeometryConstraintError(Exception):
    """Raised when input parameters produce an invalid geometry"""


class DegenerateGeometryError(Exception):
    """Raised when orienting failed because of invalid geometry"""


# Shell/offsetting logic
class SharedPointError(Exception):
    """Errors with shared points"""


class SharedPointNotFoundError(SharedPointError):
    pass


class PointNotCoincidentError(SharedPointError):
    pass


class DisconnectedChopError(SharedPointError):
    """Issued when chopping a Shell that has disconnected faces"""


### Search/retrieval
class VertexNotFoundError(Exception):
    """Raised when a vertex at a given point in space doesn't exist yet"""


class EdgeNotFoundError(Exception):
    """Raised when an edge between a given pair of vertices doesn't exist yet"""


class CornerPairError(Exception):
    """Raised when given pair of corners is not valid (for example, edge between 0 and 2)"""


class PatchNotFoundError(Exception):
    """Raised when searching for a non-existing Patch"""


### Grading
class UndefinedGradingsError(Exception):
    """Raised when the user hasn't supplied enough grading data to
    define all blocks in the mesh"""


class InconsistentGradingsError(Exception):
    """Raised when cell counts for edges on the same axis is not consistent"""


class NoInstructionError(Exception):
    """Raised when building a catalogue"""


class BlockNotFoundError(Exception):
    """Raised when building a catalogue"""


### Optimization
class NoClampError(Exception):
    """Raised when there's no junction defined for a given Clamp"""


class ClampExistsError(Exception):
    """Raised when adding a clamp to a junction that already has one defined"""


class NoCommonSidesError(Exception):
    """Raised when two cells don't share a side"""


class NoJunctionError(Exception):
    """Raised when there's a clamp defined for a vertex that doesn't exist"""


class InvalidLinkError(Exception):
    """Raised when a link has been added that doesn't connect two actual points"""


class OptimizationError(Exception):
    """Raised when optimization of a clamp produced results worse than before"""


### Mesh assembly/writing
class MeshNotAssembledError(Exception):
    """Raised when looking for assembled items on a non-assembled mesh"""
