from typing import List, Optional, Sequence

from classy_blocks.base import transforms as tr
from classy_blocks.construct.flat.sketches.annulus import Annulus
from classy_blocks.construct.flat.sketches.disk import Disk
from classy_blocks.construct.flat.sketches.sketch import Sketch
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.construct.shapes.shape import SketchedShape
from classy_blocks.types import OrientType


class RoundSolidShape(SketchedShape):
    """An object, lofted between 2 or more sketches;
    to form blocks, sketches are transformed with specified
    functions (and so are side edges) and loft operations
    are created between.

    Solid round shapes: Elbow, Frustum, Cylinder;
    they are created using an OH-grid (see Disk), have a
    'start' and 'end' sketch and an 'outer' surface."""

    def __init__(
        self,
        sketch_1: Sketch,
        sketch_2_transform: Sequence[tr.Transformation],
        sketch_mid_transform: Optional[Sequence[tr.Transformation]] = None,
    ):
        super().__init__(sketch_1, sketch_2_transform, sketch_mid_transform)

    @property
    def core(self):
        """Operations in the center of the shape"""
        return self.operations[: len(self.sketch_1.core)]

    @property
    def shell(self):
        """Operations on the outside of the shape"""
        return self.operations[len(self.sketch_1.core) :]

    def chop_radial(self, **kwargs):
        """Chop the outer 'ring', or 'shell';
        core blocks will be defined by tangential chops"""
        # scale all radial sizes to this ratio or core cells will be
        # smaller than shell's
        c2s_ratio = max(Disk.diagonal_ratio, Disk.side_ratio)
        if "start_size" in kwargs:
            kwargs["start_size"] *= c2s_ratio
        if "end_size" in kwargs:
            kwargs["end_size"] *= c2s_ratio

        self.shell[0].chop(self.radial_axis, **kwargs)

    def chop_tangential(self, **kwargs):
        """Circumferential chop; also defines core sizes"""
        for i in (0, 1, 2, -1):  # minimal number of blocks that need to be set
            self.shell[i].chop(self.tangential_axis, **kwargs)

    def set_outer_patch(self, name: str) -> None:
        for operation in self.shell:
            operation.set_patch(self.outer_patch, name)


class RoundHollowShape(RoundSolidShape):
    """An object, lofted between 2 or more sketches;
    to form blocks, sketches are transformed with specified
    functions (and so are side edges) and loft operations
    are created between.

    Hollow round shapes: ExtrudedRing (more to come)
    they are created using an OH-grid (see Disk), have a
    'start' and 'end' sketch and an 'outer' surface."""

    inner_patch: OrientType = "left"

    def __init__(
        self,
        sketch_1: Annulus,
        sketch_2_transform: List[tr.Transformation],
        sketch_mid_transform: Optional[List[tr.Transformation]] = None,
    ):
        super().__init__(sketch_1, sketch_2_transform, sketch_mid_transform)

    @property
    def shell(self) -> List[Loft]:
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
            operation.set_patch(self.inner_patch, name)
