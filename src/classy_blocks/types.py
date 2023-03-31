"""Predefined types"""
from typing import List, Union, Literal
from nptyping import NDArray, Shape, Float

# A single point can be specified as a list of floats or as a numpy array
NPPointType = NDArray[Shape["3, 1"], Float]
PointType = Union[List[float], NPPointType]
# Similar: a list of points
NPPointListType = NDArray[Shape["*, 3"], Float]
PointListType = Union[List[List[float]], NPPointListType, List[NPPointType]]
# same as PointType but with a different name to avoid confusion
NPVectorType = NPPointType
VectorType = PointType

# edge kinds as per blockMesh's definition
EdgeKindType = Literal["line", "arc", "origin", "angle", "spline", "polyLine", "project"]
# edges: arc: 1 point, projected: string, everything else: a list of points
EdgeDataType = Union[PointType, PointListType, str]

# block sides
OrientType = Literal["left", "right", "front", "back", "top", "bottom"]

AxisType = Literal[0, 1, 2]

# which block size to take when chopping
ChopTakeType = Literal["min", "max", "avg"]

# Project vertex/edge to one or multiple geometries
ProjectToType = Union[str, List[str]]
