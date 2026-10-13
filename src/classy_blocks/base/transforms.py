"""Dataclasses for packing combinations of transforms of <anything>
into an easily digestable function/method arguments"""

from dataclasses import dataclass
from typing import Optional

from classy_blocks.cbtyping import MatrixType, PointType, VectorType


@dataclass
class Transformation:
    """A superclass that addresses all
    dataclasses for transformation parameters"""


@dataclass
class Translation(Transformation):
    """Parameters required to translate an entity"""

    displacement: VectorType


@dataclass
class Rotation(Transformation):
    """Parameters required to rotate an entity"""

    axis: VectorType
    angle: float
    origin: Optional[PointType] = None


@dataclass
class Scaling(Transformation):
    """Parameters required to scale an entity"""

    ratio: float
    origin: Optional[PointType] = None


@dataclass
class Mirror(Transformation):
    """Parameters required to mirror an entity around an
    arbitrary plane"""

    normal: VectorType
    origin: Optional[PointType] = None


@dataclass
class Shear(Transformation):
    """Parameters required for a shear transform"""

    normal: VectorType
    origin: PointType

    direction: VectorType
    angle: float


@dataclass
class Matrix(Transformation):
    """An arbitrary transform, given by a homogeneous 4x4 matrix"""

    matrix: MatrixType
