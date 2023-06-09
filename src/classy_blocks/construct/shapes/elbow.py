import numpy as np

from classy_blocks.base import transforms as tr
from classy_blocks.base.exceptions import ElbowCreationError
from classy_blocks.construct.flat.sketches.disk import Disk
from classy_blocks.construct.shapes.round import RoundSolidShape
from classy_blocks.types import PointType, VectorType
from classy_blocks.util import functions as f


class Elbow(RoundSolidShape):
    """A curved round shape of varying cross-section"""

    def __init__(
        self,
        center_point_1: PointType,
        radius_point_1: PointType,
        normal_1: VectorType,
        sweep_angle: float,
        arc_center: PointType,
        rotation_axis: VectorType,
        radius_2: float,
    ):
        radius_1 = f.norm(np.asarray(radius_point_1) - np.asarray(center_point_1))
        sketch_1 = Disk(center_point_1, radius_point_1, normal_1)
        radius_ratio = radius_2 / radius_1

        transform_2 = [tr.Rotation(rotation_axis, sweep_angle, arc_center), tr.Scaling(radius_ratio)]
        transform_mid = [
            tr.Rotation(rotation_axis, sweep_angle / 2, arc_center),
            tr.Scaling((1 + radius_ratio) / 2),
        ]

        super().__init__(sketch_1, transform_2, transform_mid)

    @classmethod
    def chain(
        cls,
        source: RoundSolidShape,
        sweep_angle: float,
        arc_center: PointType,
        rotation_axis: VectorType,
        radius_2: float,
        start_face: bool = False,
    ) -> "Elbow":
        """Use another round Shape's end face as a starting point for this Elbow;
        Returns a new Elbow object. To start from the other side, use start_face = True"""
        if not isinstance(source.sketch_1, Disk):
            raise ElbowCreationError(
                "`chain()` operation failed: expecting `Disk`-type face",
                f"Given `{type(source.sketch_1)}` face instance",
            )

        if start_face:
            sketch = source.sketch_1
        else:
            sketch = source.sketch_2

        return cls(sketch.center, sketch.radius_point, sketch.normal, sweep_angle, arc_center, rotation_axis, radius_2)
