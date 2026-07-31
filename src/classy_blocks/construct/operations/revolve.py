from classy_blocks.cbtyping import VectorType
from classy_blocks.construct import edges
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.util import functions as f


class Revolve(Loft):
    """Takes a Face and revolves it by angle around axis;
    axis can be translated so that it goes through desired origin.

    Angle is given in radians,
    revolve is in positive sense (counter-clockwise - right hand rule)"""

    def __init__(self, base: Face, angle: float, axis: VectorType, origin: VectorType):
        self.base = base
        self.angle = angle
        self.axis = axis
        self.origin = origin

        bottom_face = base
        top_face = base.copy().rotate(angle, axis, origin)
        side_angle = angle

        # rotation opposite to the base face's winding creates an
        # inside-out block (the same mechanism as with mirroring)
        if f.is_loft_inverted(bottom_face.point_array, top_face.point_array):
            bottom_face, top_face = top_face, bottom_face
            side_angle = -side_angle

        super().__init__(bottom_face, top_face)

        # there are 4 side edges: the simplest is to use 'axis and angle'
        for i in range(4):
            self.add_side_edge(i, edges.Angle(side_angle, self.axis))
