from collections.abc import Sequence
from typing import Optional

from classy_blocks.base import transforms as tr
from classy_blocks.cbtyping import DirectionType, OrientType
from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.construct.flat.sketches.annulus import Annulus
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.construct.shape import LoftedShape


class RoundSolidShape(LoftedShape):
    axial_axis: DirectionType = 2  # Axis along which 'outer sides' run
    radial_axis: DirectionType = 0  # Axis that goes from center to 'outer side'
    tangential_axis: DirectionType = 1  # Axis that goes around the circumference of the shape

    start_patch: OrientType = "bottom"  # Sides of blocks that define the start patch
    end_patch: OrientType = "top"  # Sides of blocks that define the end patch"""
    outer_patch: OrientType = "right"  # Sides of blocks that define the outer surface

    def __init__(
        self,
        sketch_1: Sketch,
        sketch_2_transform: Sequence[tr.Transformation],
        sketch_mid_transform: Optional[Sequence[tr.Transformation]] = None,
    ):
        # start with sketch_1 and transform it
        # using the _transform function(transform_2_args) to obtain sketch_2;
        # use _transform function(transform_mid_args) to obtain mid sketch
        # (only if applicable)
        sketch_1 = sketch_1
        sketch_2 = sketch_1.copy().transform(sketch_2_transform)

        sketch_mid: Optional[Sketch] = None
        if sketch_mid_transform is not None:
            sketch_mid = sketch_1.copy().transform(sketch_mid_transform)

        super().__init__(sketch_1, sketch_2, sketch_mid)

    def chop_axial(self, **kwargs):
        """Chop the shape between start and end face"""
        super().chop(self.axial_axis, **kwargs)

    def chop_radial(self, **kwargs):
        """Chop the outer 'ring', or 'shell';
        core blocks will be defined by tangential chops"""
        # scale all radial sizes to this ratio or core cells will be
        # smaller than shell's
        c2s_ratio = max(self.sketch_1.diagonal_ratio, self.sketch_1.core_ratio)
        if "start_size" in kwargs:
            kwargs["start_size"] *= c2s_ratio
        if "end_size" in kwargs:
            kwargs["end_size"] *= c2s_ratio

        super().chop(self.radial_axis, **kwargs)

    def chop_tangential(self, **kwargs):
        """Circumferential chop; also defines core sizes"""
        super().chop(self.tangential_axis, **kwargs)

    @property
    def core(self):
        """Operations in the center of the shape"""
        return self.operations[: len(self.sketch_1.core)]

    @property
    def shell(self):
        """Operations on the outside of the shape"""
        return self.operations[len(self.sketch_1.core) :]

    def set_outer_patch(self, name: str) -> None:
        for operation in self.shell:
            operation.set_patch(self.outer_patch, name)

    def remove_inner_edges(self, start: bool = True, end: bool = True) -> None:
        """Removes spline edges from cylinders.
        This needs to be done in cases where any of the start/end plane points will move
        (due to optimization or manual adjustments)."""
        if start:
            for face in self.sketch_1.core:
                face.remove_edges()

        if end:
            for face in self.sketch_2.core:
                face.remove_edges()


class RoundHollowShape(RoundSolidShape):
    def __init__(
        self,
        sketch_1: Annulus,
        sketch_2_transform: list[tr.Transformation],
        sketch_mid_transform: Optional[list[tr.Transformation]] = None,
    ):
        super().__init__(sketch_1, sketch_2_transform, sketch_mid_transform)

    @property
    def shell(self) -> list[Loft]:
        """The 'outer' (that is, 'all') operations"""
        return self.operations

    def chop_tangential(self, **kwargs) -> None:
        """Circumferential chop"""
        # Ring has no 'core' so tangential chops must be defined explicitly
        for operation in self.shell:
            operation.chop(self.tangential_axis, **kwargs)

    def chop_radial(self, **kwargs):
        """Chop the outer 'ring', or 'shell'"""
        self.shell[0].chop(self.radial_axis, **kwargs)

    def set_inner_patch(self, name: str) -> None:
        """Assign the faces of inside surface to a named patch"""
        for operation in self.shell:
            operation.set_patch("left", name)
