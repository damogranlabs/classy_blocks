""" Defines operations (Loft, Extrude, Revolve, Wedge).

Analogous to a sketch in 3D CAD software,
a Face is a collection of 4 vertices and 4 edges.
An operation is a 3D shape obtained by swiping a Face
into 3rd dimension depending on the Operation. """
from typing import List, Union, Optional, TypeVar

import numpy as np

from classy_blocks import types

from classy_blocks.data.data_object_base import DataObjectBase

from classy_blocks.data.block_data import BlockData
from classy_blocks.construct.flat.face import Face
from classy_blocks.util import functions as f


class Wedge(Revolve):
    """Revolves 'face' around x-axis symetrically by +/- angle/2.
    By default, the angle is 2 degrees.

    Used for creating wedge-type geometries for axisymmetric cases.
    Automatically creates wedge patches* (you still
    need to include them in changeDictionaryDict - type: wedge).

    * - default naming of block sides is not very intuitive
    for wedge geometry so additional methods are available for wedges:
        set_outer_patch,
        set_inner_patch,
        set_left_patch,
        set_right_patch,
    other two patches are wedge_left and wedge_right. Sides are named
    according to this sketch:

                        outer
        _________________________________
        |                               |
        | left                    right |
        |_______________________________|
                        inner
    __  _____  __  _____  __  _____  __  __ axis of symmetry (x)"""

    def __init__(self, face: Face, angle=f.deg2rad(2)):
        # default axis
        axis = [1, 0, 0]
        # default origin
        origin = [0, 0, 0]

        # first, rotate this face forward, then use init this as Revolve
        # and rotate the same face
        base = face.rotate(axis, -angle / 2, origin)

        super().__init__(base, angle, axis, origin)

        # assign 'wedge_left' and 'wedge_right' patches
        super().set_patch("top", "wedge_front")
        super().set_patch("bottom", "wedge_back")

        # there's also only 1 cell in z-direction
        self.chop(2, count=1)

    def set_patch(self, *args, **kwargs):
        raise NotImplementedError("Use set_[outer|inner|left|right]_patch methods for wedges")

    def set_outer_patch(self, patch_name: str) -> None:
        """Sets the patch away from the wall (see sketch in class definition)"""
        super().set_patch("back", patch_name)

    def set_inner_patch(self, patch_name: str) -> None:
        """Sets the patch closest to the wall (see sketch in class definition)"""
        super().set_patch("front", patch_name)

    def set_left_patch(self, patch_name: str) -> None:
        """Sets the patch, encountered first when rotating in a positive sense
        (see sketch in class definition)"""
        super().set_patch("left", patch_name)

    def set_right_patch(self, patch_name: str) -> None:
        """Sets the patch, encountered last when rotating in a positive sense
        (see sketch in class definition)"""
        super().set_patch("right", patch_name)
