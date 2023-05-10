from typing import Optional

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.revolve import Revolve
from classy_blocks.util import functions as f


class Wedge(Revolve):
    """Revolves 'face' around x-axis symetrically by +/- angle/2.
    By default, the angle is 2 degrees.

    Used for creating wedge-type geometries for axisymmetric cases.
    Automatically creates wedge patches* (you still
    need to include them in changeDictionaryDict - type: wedge).

    * - default naming of block sides is not very intuitive
    for wedge geometry so additional names are available for wedges:
        set_inner_patch() (= 'front')
        set_outer_patch() (= 'back')
    other two patches are wedge_left and wedge_right. Sides are named
    according to this sketch:

                        outer
        _________________________________
        |                               |
        | left                    right |
        |_______________________________|
                        inner
    __  _____  __  _____  __  _____  __  __ axis of symmetry (x)"""

    def __init__(self, face: Face, angle: Optional[float] = None):
        if angle is None:
            angle = f.deg2rad(2)
        # default axis
        axis = [1.0, 0.0, 0.0]
        # default origin
        origin = [0.0, 0.0, 0.0]

        # first, rotate this face forward, then use init this as Revolve
        # and rotate the same face
        base = face.copy().rotate(-angle / 2, axis, origin)

        super().__init__(base, angle, axis, origin)

        # assign 'wedge_left' and 'wedge_right' patches
        super().set_patch("top", "wedge_front")
        super().set_patch("bottom", "wedge_back")

        # there's also only 1 cell in z-direction
        self.chop(2, count=1)

    def set_inner_patch(self, name: str) -> None:
        """Set patch closest to axis of rotation (x)"""
        super().set_patch("front", name)

    def set_outer_patch(self, name: str) -> None:
        """Set patch away from axis of rotation (x)"""
        super().set_patch("back", name)
