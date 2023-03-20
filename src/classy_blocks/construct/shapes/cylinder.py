import numpy as np

from classy_blocks.types import PointType
from classy_blocks.construct.flat.disk import Disk
from classy_blocks.construct.shapes.round import RoundShape
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

    @classmethod
    def chain(cls, source:RoundShape, length:float) -> 'Cylinder':
        """Creates a new Cylinder on start or end face of a round Shape (Elbow, Frustum, Cylinder);
        Use length > 0 to extrude 'forward' from source's end face;
        Use length < 0 to extrude 'backward' from source' start face"""
        assert source.sketch_class == Disk
        # TODO: TEST
        if length > 0:
            sketch = source.sketch_2
        else:
            sketch = source.sketch_1

        axis_point_1 = sketch.center_point
        radius_point_1 = sketch.radius_point
        normal = sketch.normal

        axis_point_2 = axis_point_1 + f.unit_vector(normal) * length

        return cls(axis_point_1, axis_point_2, radius_point_1)
