from typing import List, Union

import numpy as np

from classy_blocks.base import transforms as tr
from classy_blocks.base.exceptions import ExtrudedRingCreationError
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.flat.sketches.annulus import Annulus
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.construct.operations.revolve import Revolve
from classy_blocks.construct.shapes.round import RoundHollowShape, RoundSolidShape
from classy_blocks.types import OrientType, PointType
from classy_blocks.util import functions as f


class ExtrudedRing(RoundHollowShape):
    """A ring, created by specifying its base, then extruding it"""

    def __init__(
        self,
        axis_point_1: PointType,
        axis_point_2: PointType,
        outer_radius_point_1: PointType,
        inner_radius: float,
        n_segments: int = 8,
    ):
        axis = np.asarray(axis_point_2) - np.asarray(axis_point_1)

        super().__init__(
            Annulus(axis_point_1, outer_radius_point_1, axis, inner_radius, n_segments),
            [tr.Translation(axis)],
            None,
        )

    @classmethod
    def chain(cls, source: "ExtrudedRing", length: float, start_face: bool = False) -> "ExtrudedRing":
        """Creates a new ExtrudedRing on end face of source ring;
        use start_face=False to chain 'backwards' from the first face"""
        if length < 0:
            raise ExtrudedRingCreationError(
                "`chain()` operation failed: use a positive length and `start_face=True` to chain 'backwards'",
                f"Given length: {length}, `start_face={start_face}`",
            )

        if start_face:
            sketch = source.sketch_1
            length = -length
        else:
            sketch = source.sketch_2

        return cls(
            sketch.center,
            sketch.center + f.unit_vector(sketch.normal) * length,
            sketch.outer_radius_point,
            sketch.inner_radius,
            n_segments=sketch.n_segments,
        )

    @classmethod
    def expand(cls, source: Union[RoundSolidShape, RoundHollowShape], thickness: float) -> "ExtrudedRing":
        """Create a new concentric Ring with radius, enlarged by 'thickness';
        Can be used on Cylinder or ExtrudedRing"""
        sketch_1 = source.sketch_1
        sketch_2 = source.sketch_2

        new_radius_point = sketch_1.center + f.unit_vector(sketch_1.radius_point - sketch_1.center) * (
            sketch_1.radius + thickness
        )

        return cls(sketch_1.center, sketch_2.center, new_radius_point, sketch_1.radius, n_segments=sketch_1.n_segments)

    @classmethod
    def contract(cls, source: "ExtrudedRing", inner_radius: float) -> "ExtrudedRing":
        """Create a new ring on inner surface of the source"""
        if inner_radius <= 0:
            raise ExtrudedRingCreationError(
                "Unable to perform `contract()` operation for inner radius < 0: use `Cylinder.fill(extruded_ring)`",
                f"Inner radius: {inner_radius}",
            )

        sketch_1 = source.sketch_1
        sketch_2 = source.sketch_2
        if inner_radius > sketch_1.inner_radius:
            raise ExtrudedRingCreationError(
                "Unable to perform `contract()` operation: new inner radius must be smaller than source's",
                f"Inner radius: {inner_radius}, sketch inner radius: {sketch_1.inner_radius}",
            )

        return cls(
            sketch_1.center, sketch_2.center, sketch_1.inner_radius_point, inner_radius, n_segments=sketch_1.n_segments
        )


class RevolvedRing(ExtrudedRing):
    """A ring specified by its cross-section; can be of arbitrary shape.
    Face points must be specified in the following order:
            p3---___
           /        ---p2
          /              \\
         p0---------------p1

    0---- -- ----- -- ----- -- ----- -- --->> axis

    In this case, chop_*() will work as intended, otherwise
    the axes will be swapped or blocks will be inverted.

    Because of RevolvedRing's arbitrary shape, there is no
    'start' or 'end' sketch and .expand()/.contract() methods
    are not available.

    This shape is useful when building more complex shapes
    of revolution (with non-orthogonal blocks)
    from known 2d-blocking in cross-section."""

    # TODO: automatic point sorting to match ExtrudedRing numbering?

    axial_axis = 0
    radial_axis = 1
    tangential_axis = 2

    start_patch: OrientType = "left"
    end_patch: OrientType = "right"
    inner_patch: OrientType = "front"
    outer_patch: OrientType = "back"

    def __init__(self, axis_point_1: PointType, axis_point_2: PointType, cross_section: Face, n_segments: int = 8):
        self.axis_point_1 = np.asarray(axis_point_1)
        self.axis_point_2 = np.asarray(axis_point_2)
        self.axis = self.axis_point_2 - self.axis_point_1
        self.center_point = self.axis_point_1

        angle = 2 * np.pi / n_segments

        revolve = Revolve(cross_section, angle, self.axis, self.axis_point_1)

        self.revolves: List[Operation] = [
            revolve.copy().rotate(i * angle, self.axis, self.center_point) for i in range(n_segments)
        ]

    def set_inner_patch(self, name: str) -> None:
        """Assign the faces of inside surface to a named patch"""
        for operation in self.operations:
            operation.set_patch(self.inner_patch, name)

    # methods/properties that differ from a lofted-sketch type of shape
    @property
    def operations(self) -> List[Operation]:
        return self.revolves
