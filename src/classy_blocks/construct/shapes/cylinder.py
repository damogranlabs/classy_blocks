from typing import ClassVar, Union

import numpy as np

from classy_blocks.base import transforms as tr
from classy_blocks.base.exceptions import CylinderCreationError
from classy_blocks.cbtyping import PointType
from classy_blocks.construct.flat.sketches.disk import Disk, HalfDisk, QuarterDisk
from classy_blocks.construct.shapes.rings import ExtrudedRing
from classy_blocks.construct.shapes.round import RoundSolidShape
from classy_blocks.util import functions as f
from classy_blocks.util.constants import TOL


class SemiCylinder(RoundSolidShape):
    """Half of a cylinder; it is constructed from
    given point and axis in a positive sense - right-hand rule.

    Args:
    axis_point_1: position of start face
    axis_point_2: position of end face
    radius_point_1: defines starting point and radius"""

    sketch_class: ClassVar[Union[type[Disk], type[HalfDisk], type[QuarterDisk]]] = HalfDisk

    def __init__(self, axis_point_1: PointType, axis_point_2: PointType, radius_point_1: PointType):
        axis_point_1 = np.asarray(axis_point_1)
        axis = np.asarray(axis_point_2) - axis_point_1
        radius_point_1 = np.asarray(radius_point_1)

        diff = np.dot(axis, radius_point_1 - axis_point_1)
        if diff > TOL:
            raise CylinderCreationError(
                "Axis and radius vectors are not perpendicular", f"Difference: {diff}, tolerance: {TOL}"
            )

        transform_2: list[tr.Transformation] = [tr.Translation(axis)]

        super().__init__(self.sketch_class(axis_point_1, radius_point_1, axis), transform_2, None)

    def set_symmetry_patch(self, name: str) -> None:
        self.shell[0].set_patch("front", name)
        self.shell[3].set_patch("back", name)
        self.core[0].set_patch("front", name)
        self.core[1].set_patch("front", name)


class QuarterCylinder(RoundSolidShape):
    """Quarter of a cylinder; it is constructed from
    given point and axis in a positive sense - right-hand rule.

    Args:
    axis_point_1: position of start face
    axis_point_2: position of end face
    radius_point_1: defines starting point and radius"""

    sketch_class: ClassVar[Union[type[Disk], type[HalfDisk], type[QuarterDisk]]] = QuarterDisk

    def __init__(self, axis_point_1: PointType, axis_point_2: PointType, radius_point_1: PointType):
        axis_point_1 = np.asarray(axis_point_1)
        axis = np.asarray(axis_point_2) - axis_point_1
        radius_point_1 = np.asarray(radius_point_1)

        diff = np.dot(axis, radius_point_1 - axis_point_1)
        if diff > TOL:
            raise CylinderCreationError(
                "Axis and radius vectors are not perpendicular", f"Difference: {diff}, tolerance: {TOL}"
            )

        transform_2: list[tr.Transformation] = [tr.Translation(axis)]

        super().__init__(self.sketch_class(axis_point_1, radius_point_1, axis), transform_2, None)

    def set_symmetry_patch(self, name: str) -> None:
        self.shell[0].set_patch("front", name)
        self.shell[1].set_patch("back", name)
        self.core[0].set_patch("front", name)
        self.core[0].set_patch("left", name)


class Cylinder(SemiCylinder):
    sketch_class = Disk

    @classmethod
    def chain(cls, source: RoundSolidShape, length: float, start_face: bool = False) -> "Cylinder":
        """Creates a new Cylinder on start or end face of a round Shape (Elbow, Frustum, Cylinder);
        Use length > 0 to extrude 'forward' from source's end face;
        Use length > 0 and `start_face=True` to extrude 'backward' from source's start face"""
        if length < 0:
            raise CylinderCreationError(
                "`chain()` operation failed: use a positive length and `start_face=True` to chain 'backwards'",
                f"Given length: {length}, `start_face={start_face}`",
            )

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
        if source.sketch_1.n_segments != 8:
            raise CylinderCreationError(
                "`chain()` operation failed: nly rings made from 8 segments can be filled",
                f"{source.sketch_1.n_segments} segments available",
            )

        return cls(source.sketch_1.center, source.sketch_2.center, source.sketch_1.inner_radius_point)

    def set_symmetry_patch(self, _name: str) -> None:
        raise RuntimeError("Cylinder does not have symmetry patches")
