import numpy as np

from classy_blocks.types import PointType
from classy_blocks.construct.shapes.frustum import Frustum

from classy_blocks.util import functions as f

class Cylinder(Frustum):
    """A Cylinder.
    
    Args:
    axis_point_1: position of start face
    axis_point_2: position of end face
    radius_point_1: defines starting point and radius"""
    def __init__(self, axis_point_1:PointType, axis_point_2:PointType, radius_point_1:PointType):
        radius_1 = f.norm(np.array(radius_point_1) - np.array(axis_point_1))

        super().__init__(axis_point_1, axis_point_2, radius_point_1, radius_1)
