"""Predefined types"""
from typing import TypeAlias, List, Union, Literal
from numpy.typing import ArrayLike

# A single point can be specified as a list of floats or as a numpy array
PointType:TypeAlias = Union[List[float], ArrayLike]
# Similar: a list of points 
PointListType:TypeAlias = Union[List[PointType], ArrayLike]
# same as PointType but with a different name to avoid confusion
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
