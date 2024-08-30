from typing import Any, Callable, List, Literal, Sequence, TypedDict, Union

from nptyping import NDArray, Shape

# A plain list of floats
FloatListType = NDArray[Shape["1, *"], Any]

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

AxisType = Literal[0, 1, 2]

# the complete guide to chopping
ChopPreserveType = Literal["start_size", "end_size", "c2c_expansion"]


class ChopArgs(TypedDict, total=False):
    length_ratio: float
    start_size: float
    c2c_expansion: float
    count: int
    end_size: float
    total_expansion: float
    preserve: ChopPreserveType


# Project vertex/edge to one or multiple geometries
ProjectToType = Union[str, List[str]]

# A list of indexes that define a quad
IndexType = List[int]
