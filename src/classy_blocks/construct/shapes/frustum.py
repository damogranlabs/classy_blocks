from typing import Optional

import numpy as np


from classy_blocks.types import PointType
from classy_blocks.construct.flat.disk import Disk
from classy_blocks.construct.shapes.round import RoundShape

from classy_blocks.util import functions as f

class Frustum(RoundShape):
    """Creates a cone frustum (truncated cylinder).

    Args:
        axis_point_1: position of the starting face and axis start point
        axis_point_2: position of the end face and axis end point
        radius_point_1: defines starting point for blocks
        radius_2: defines end radius; NOT A POINT!

        Sides are straight unless radius_mid is given; in that case a profiled body
        of revolution is created."""
    def transform_function(self, **kwargs):
        new_sketch = self.sketch_1.copy()
        new_sketch.translate(kwargs['displacement'])
        new_sketch.scale(kwargs['radius']/self.sketch_1.radius)

        return new_sketch

    def __init__(self,
                 axis_point_1:PointType,
                 axis_point_2:PointType,
                 radius_point_1:PointType,
                 radius_2: float, radius_mid:Optional[float]=None):
        self.axis = np.asarray(axis_point_2) - np.asarray(axis_point_1)

        if radius_mid is None:
            mid_params = None
        else:
            mid_params = {"displacement": self.axis / 2, "radius": radius_mid}

        super().__init__(
            [axis_point_1, radius_point_1, self.axis],
            {"displacement": self.axis, "radius": radius_2},
            mid_params
        )

    
    @classmethod
    def chain(cls, source, length, radius_2, radius_mid=None):
        """Chain this Frustum to an existing Shape;
        Use length < 0 to begin on start face and go 'backwards'"""
        # TODO: TEST
        assert source.sketch_class == Disk

        if length < 0:
            sketch = source.sketch_1
        else:
            sketch = source.sketch_2

        axis_point_2 = sketch.center_point + f.unit_vector(sketch.normal) * length

        return cls(sketch.center_point, axis_point_2, sketch.radius_point, radius_2, radius_mid)
