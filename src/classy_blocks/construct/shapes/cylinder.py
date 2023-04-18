from typing import List

import numpy as np

from classy_blocks.types import PointType
from classy_blocks.base import transforms as tr
from classy_blocks.construct.flat.sketches.disk import Disk
from classy_blocks.construct.shapes.round import RoundSolidShape
from classy_blocks.construct.shapes.rings import ExtrudedRing
from classy_blocks.util import functions as f
from classy_blocks.util.constants import TOL


class Cylinder(RoundSolidShape):
    """A Cylinder.

    Args:
    axis_point_1: position of start face
    axis_point_2: position of end face
    radius_point_1: defines starting point and radius"""

    def __init__(self, axis_point_1: PointType, axis_point_2: PointType, radius_point_1: PointType):
        axis_point_1 = np.asarray(axis_point_1)
        axis = np.asarray(axis_point_2) - axis_point_1
        radius_point_1 = np.asarray(radius_point_1)

        # TODO: TEST
        assert np.dot(axis, radius_point_1 - axis_point_1) < TOL, "Make sure axis and radius vectors are perpendicular"

        transform_2: List[tr.Transformation] = [tr.Translation(axis)]

        super().__init__(Disk(axis_point_1, radius_point_1, axis), transform_2, None)

    @classmethod
    def chain(cls, source: RoundSolidShape, length: float, start_face: bool = False) -> "Cylinder":
        """Creates a new Cylinder on start or end face of a round Shape (Elbow, Frustum, Cylinder);
        Use length > 0 to extrude 'forward' from source's end face;
        Use length < 0 to extrude 'backward' from source' start face"""
        assert length > 0, "Use a positive length and start_face=True to chain 'backwards'"

        if start_face:
            sketch = source.sketch_1
            length = -length
        else:
            sketch = source.sketch_2

        axis_point_1 = sketch.center
        radius_point_1 = sketch.radius_point
        normal = sketch.normal

        axis_point_2 = axis_point_1 + f.unit_vector(normal) * length

        return cls(axis_point_1, axis_point_2, radius_point_1)

    @classmethod
    def fill(cls, source: "ExtrudedRing") -> "Cylinder":
        """Fills the inside of the ring with a matching cylinder"""
        assert source.sketch_1.n_segments == 8, "Only rings made from 8 segments can be filled"

        return cls(source.sketch_1.center, source.sketch_2.center, source.sketch_1.inner_radius_point)
