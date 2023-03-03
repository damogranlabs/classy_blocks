"""Predefined types"""
from typing import TypeAlias, List, Union, Literal
from nptyping import NDArray, Shape, Float

# A single point can be specified as a list of floats or as a numpy array
NPPointType = NDArray[Shape["3, 1"], Float]
PointType:TypeAlias = Union[List[float], NPPointType]
# Similar: a list of points
NPPointListType = NDArray[Shape["*, 3"], Float]
PointListType:TypeAlias = Union[List[List[float]], NPPointListType]
# same as PointType but with a different name to avoid confusion
NPVectorType = NPPointType
VectorType:TypeAlias = PointType

# edge kinds as per blockMesh's definition
EdgeKindType:TypeAlias = Literal['line', 'arc', 'origin', 'angle', 'spline', 'polyLine', 'project']
# edges: arc: 1 point, projected: string, everything else: a list of points
EdgeDataType:TypeAlias = Union[PointType, PointListType, str]

# block sides
OrientType:TypeAlias = Literal['left', 'right', 'front', 'back', 'top', 'bottom']

AxisType:TypeAlias = Literal[0, 1, 2]

# which block size to take when chopping
ChopTakeType:TypeAlias = Literal['min', 'max', 'avg']
