"""Predefined types"""
from typing import List, Union, Literal
from numpy.typing import ArrayLike

# A single point can be specified as a list of floats or as a numpy array
PointType = Union[List[float], ArrayLike]
# Similar: a list of points 
PointListType = Union[List[PointType], ArrayLike]
# same as PointType but with a different name to avoid confusion
VectorType = PointType

# edge kinds as per blockMesh's definition
EdgeKindType = Literal['arc', 'origin', 'angle', 'spline', 'polyLine', 'project']
# edges: arc: 1 point, projected: string, everything else: a list of points
EdgeDataType = Union[PointType, PointListType, str]

# block sides
OrientType = Literal['left, right', 'front', 'back', 'top', 'bottom']
