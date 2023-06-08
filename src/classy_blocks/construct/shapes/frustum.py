from typing import Optional

import numpy as np

from classy_blocks.base import transforms as tr
from classy_blocks.base.exceptions import FrustumCreationError
from classy_blocks.construct.flat.sketches.disk import Disk
from classy_blocks.construct.shapes.round import RoundSolidShape
from classy_blocks.types import PointType
from classy_blocks.util import functions as f
from classy_blocks.util.constants import TOL


class Frustum(RoundSolidShape):
    """Creates a cone frustum (truncated cylinder).

    Args:
        axis_point_1: position of the starting face and axis start point
        axis_point_2: position of the end face and axis end point
        radius_point_1: defines starting point for blocks
        radius_2: defines end radius; NOT A POINT!

        Sides are straight unless radius_mid is given; in that case a profiled body
        of revolution is created."""

    sketch_class = Disk

    def __init__(
        self,
        axis_point_1: PointType,
        axis_point_2: PointType,
        radius_point_1: PointType,
        radius_2: float,
        radius_mid: Optional[float] = None,
    ):
        axis_point_1 = np.asarray(axis_point_1)
        axis = np.asarray(axis_point_2) - axis_point_1
        radius_point_1 = np.asarray(radius_point_1)

        radius_vector_1 = radius_point_1 - axis_point_1
        radius_1 = f.norm(radius_vector_1)

        # TODO: TEST
        diff = np.dot(axis, radius_vector_1)
        if diff > TOL:
            raise FrustumCreationError(
                "Axis and radius vectors are not perpendicular", f"Difference: {diff}, tolerance: {TOL}"
            )

        transform_2 = [tr.Translation(axis_point_2 - axis_point_1), tr.Scaling(radius_2 / radius_1)]

        if radius_mid is None:
            transform_mid = None
        else:
            transform_mid = [tr.Translation(axis / 2), tr.Scaling(radius_mid / radius_1)]

        super().__init__(Disk(axis_point_1, radius_point_1, axis), transform_2, transform_mid)

    @classmethod
    def chain(
        cls,
        source: RoundSolidShape,
        length: float,
        radius_2: float,
        start_face: bool = False,
        radius_mid: Optional[float] = None,
    ) -> "Frustum":
        """Chain this Frustum to an existing Shape;
        Use length > 0 to begin on source's end face;
        Use length > 0 and `start_face=True` to begin on source's start face and go backwards
        """
        if length < 0:
            raise FrustumCreationError(
                "`chain()` operation failed: use a positive length and `start_face=True` to chain 'backwards'",
                f"Given length: {length}, `start_face={start_face}`",
            )

        if start_face:
            sketch = source.sketch_1
            length = -length
        else:
            sketch = source.sketch_2

        axis_point_2 = sketch.center + f.unit_vector(sketch.normal) * length

        return cls(sketch.center, axis_point_2, sketch.radius_point, radius_2, radius_mid)
