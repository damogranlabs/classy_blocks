from collections.abc import Sequence
from typing import Any, Callable, Literal, Optional, TypedDict, Union

from nptyping import Float, NDArray, Shape

# A plain list of floats
FloatListType = NDArray[Shape["1, *"], Float]

# A single point can be specified as a list of floats or as a numpy array
NPPointType = NDArray[Shape["3, 1"], Any]
PointType = Union[Sequence[Union[int, float]], NPPointType]
# Similar: a list of points
NPPointListType = NDArray[Shape["*, 3"], Any]
PointListType = Union[NPPointListType, Sequence[PointType], Sequence[NPPointType]]
# same as PointType but with a different name to avoid confusion
NPVectorType = NPPointType
VectorType = PointType

# parametric curve
ParamCurveFuncType = Callable[[float], NPPointType]

# edge kinds as per blockMesh's definition
EdgeKindType = Literal["line", "arc", "origin", "angle", "spline", "polyLine", "project", "curve"]
# edges: arc: 1 point, projected: string, everything else: a list of points
EdgeDataType = Union[PointType, PointListType, str]

# block sides
OrientType = Literal["left", "right", "front", "back", "top", "bottom"]
DirectionType = Literal[0, 1, 2]

# Project vertex/edge to one or multiple geometries
ProjectToType = Union[str, list[str]]

# A list of indexes that define a quad
IndexType = list[int]


# the complete guide to chopping
ChopTakeType = Literal["min", "max", "avg"]  # which wire of the block to take as reference length
ChopPreserveType = Literal["start_size", "end_size", "c2c_expansion", "total_expansion"]  # what value to keep


class ChopArgs(TypedDict, total=False):
    """All chopping parameters"""

    length_ratio: float
    start_size: float
    c2c_expansion: float
    count: int
    end_size: float
    total_expansion: float
    take: ChopTakeType
    preserve: ChopPreserveType


# what goes into blockMeshDict's block grading specification
GradingSpecType = tuple[float, int, float]
# Used by autograders
CellSizeType = Optional[float]

# Geometry definition
GeometryType = dict[str, list[str]]
