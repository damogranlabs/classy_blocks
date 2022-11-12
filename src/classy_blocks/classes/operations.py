""" Defines operations (Loft, Extrude, Revolve, Wedge).

Analogous to a sketch in 3D CAD software,
a Face is a collection of 4 vertices and 4 edges.
An operation is a 3D shape obtained by swiping a Face
into 3rd dimension depending on the Operation. """
from typing import List, Union, NoReturn, TypeVar

import numpy as np

from classy_blocks.classes.block import Block
from classy_blocks.classes.primitives import Edge
from classy_blocks.classes.primitives import transform_edges
from classy_blocks.classes.flat.face import Face
from classy_blocks.util import functions as f

Op = TypeVar("Op", bound="Operation")


class Operation:
    """Base of all other operations"""

    def __init__(self, bottom_face: Face, top_face: Face, side_edges: List[Edge] = None):
        self.bottom_face = bottom_face
        self.top_face = top_face

        # edges between vertices on same face
        self.edges = self.bottom_face.get_edges() + self.top_face.get_edges(top_face=True)

        # edges between same vertices between faces
        self.side_edges = side_edges

        if self.side_edges is not None:
            if len(self.side_edges) != 4:
                raise Exception("Provide 4 edges for sides; use None for straight lines")

            for i in range(4):
                e = self.side_edges[i]

                if e is not None:
                    self.edges.append(Edge(i, i + 4, e))

        # create a block and edges
        self.block = Block.create_from_points(np.concatenate((bottom_face.points, top_face.points)), self.edges)

    def chop(self, axis: int, **kwargs) -> NoReturn:
        """Chop the operation (count/grading) in given axis:
        0: along first edge of a face
        1: along second edge of a face
        2: between faces / along operation path
        """
        self.block.chop(axis, **kwargs)

    def set_patch(self, sides: Union[str, List[str]], patch_name: str) -> NoReturn:
        """bottom: bottom face
        top: top face

        front: along first edge of a face
        back: opposite front

        right: along second edge of a face
        left: opposite right"""
        self.block.set_patch(sides, patch_name)

    def translate(self: Op, vector: List) -> Op:
        """returns a translated copy of this Operation"""
        vector = np.array(vector)

        bottom_face = self.bottom_face.translate(vector)
        top_face = self.top_face.translate(vector)

        side_edges = transform_edges(self.side_edges, lambda v: v + vector)

        return Operation(bottom_face, top_face, side_edges)

    def rotate(self: Op, axis: List, angle: float, origin: List = None) -> Op:
        """Copies this Operation and rotates it around an arbitrary axis and origin.
        The original Operation stays in place."""
        if origin is None:
            origin = [0, 0, 0]

        axis = np.array(axis)
        origin = np.array(origin)

        bottom_face = self.bottom_face.rotate(axis, angle, origin)
        top_face = self.top_face.rotate(axis, angle, origin)

        side_edges = transform_edges(self.side_edges, lambda point: f.arbitrary_rotation(point, axis, angle, origin))

        return Operation(bottom_face, top_face, side_edges)

    def set_cell_zone(self, cell_zone: str) -> NoReturn:
        """Assign a cellZone to this block."""
        self.block.cell_zone = cell_zone


class Loft(Operation):
    """since any possible block shape can be created with Loft operation,
    Loft is the most low-level of all operations. Anything included in Loft
    must also be included in Operation."""

    pass


class Extrude(Loft):
    """Takes a Face and extrudes it in given extrude_direction"""

    def __init__(self, base: Face, extrude_vector: list):
        self.base = base
        self.extrude_vector = extrude_vector

        top_face = base.translate(self.extrude_vector)

        super().__init__(base, top_face)


class Revolve(Loft):
    """Takes a Face and revolves it by angle around axis;
    axis can be translated so that it goes through desired origin.

    Angle is given in radians,
    revolve is in positive sense (counter-clockwise)"""

    def __init__(self, base: Face, angle: list, axis: list, origin: list):
        self.base = base
        self.angle = angle
        self.axis = axis
        self.origin = origin

        bottom_face = base
        top_face = base.rotate(axis, angle, origin)

        # there are 4 side edges: rotate each vertex of bottom_face
        # by angle/2
        side_edges = [f.arbitrary_rotation(p, self.axis, self.angle / 2, self.origin) for p in self.base.points]

        super().__init__(bottom_face, top_face, side_edges)


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

    def set_outer_patch(self, patch_name: str) -> NoReturn:
        """Sets the patch away from the wall (see sketch in class definition)"""
        super().set_patch("back", patch_name)

    def set_inner_patch(self, patch_name: str) -> NoReturn:
        """Sets the patch closest to the wall (see sketch in class definition)"""
        super().set_patch("front", patch_name)

    def set_left_patch(self, patch_name: str) -> NoReturn:
        """Sets the patch, encountered first when rotating in a positive sense
        (see sketch in class definition)"""
        super().set_patch("left", patch_name)

    def set_right_patch(self, patch_name: str) -> NoReturn:
        """Sets the patch, encountered last when rotating in a positive sense
        (see sketch in class definition)"""
        super().set_patch("right", patch_name)
