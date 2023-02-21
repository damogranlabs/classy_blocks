""" Defines operations (Loft, Extrude, Revolve, Wedge).

Analogous to a sketch in 3D CAD software,
a Face is a collection of 4 vertices and 4 edges.
An operation is a 3D shape obtained by swiping a Face
into 3rd dimension depending on the Operation. """
from typing import List, Union, Optional, TypeVar

import numpy as np

from classy_blocks import types

from classy_blocks.base import ObjectBase

from classy_blocks.data.block_data import BlockData
from classy_blocks.construct.flat.face import Face
from classy_blocks.util import functions as f

Op = TypeVar("Op", bound="Operation")


class Operation(ObjectBase):
    """Base of all other operations"""
    def __init__(self, bottom_face: Face, top_face: Face):
        self.bottom_face = bottom_face
        self.top_face = top_face

        # create a block and assign edges to it
        self.block = BlockData(np.concatenate((bottom_face.points, top_face.points)))
        #self.block.edges += bottom_face.get_edges(False)
        #self.block.edges += top_face.get_edges(True)

        #self.side_edges = []

    def add_side_edge(self, index:int, points, kind:Optional[types.EdgeKindType]=None):
        """Add an edge between two vertices at the same
        corner of the lower and upper face (index and index+4 or vice versa)."""
        index = index % 4
        self.side_edges.append(self.block.add_edge(index, index+4, points, kind=kind))

    def chop(self, axis: int, **kwargs) -> None:
        """Chop the operation (count/grading) in given axis:
        0: along first edge of a face
        1: along second edge of a face
        2: between faces / along operation path
        """
        self.block.chop(axis, **kwargs)

    def set_patch(self, sides: Union[str, List[str]], patch_name: str) -> None:
        """bottom: bottom face
        top: top face

        front: along first edge of a face
        back: opposite front

        right: along second edge of a face
        left: opposite right"""
        self.block.set_patch(sides, patch_name)

    def translate(self: Op, vector: List) -> Op:
        """returns a translated copy of this Operation"""
        vector = np.asarray(vector)

        bottom_face = self.bottom_face.translate(vector)
        top_face = self.top_face.translate(vector)

        new_op = Operation(bottom_face, top_face)
        for edge in self.side_edges:
            new_op.add_side_edge(edge.translate(vector))

        return new_op

    def rotate(self: Op, axis: List, angle: float, origin: List = None) -> Op:
        """Copies this Operation and rotates it around an arbitrary axis and origin.
        The original Operation stays in place."""
        if origin is None:
            origin = [0, 0, 0]

        axis = np.asarray(axis)
        origin = np.asarray(origin)

        bottom_face = self.bottom_face.rotate(axis, angle, origin)
        top_face = self.top_face.rotate(axis, angle, origin)

        new_op = Operation(bottom_face, top_face)
        for edge in self.side_edges:
            new_op.add_side_edge(edge.rotate(axis, angle, origin))

        return new_op

    # TODO: operation.scale?

    def set_cell_zone(self, cell_zone: str) -> None:
        """Assign a cellZone to this block."""
        self.block.cell_zone = cell_zone

    @property
    def blocks(self) -> List[BlockData]:
        return [self.block]

    @property
    def edges(self):
        return self.block.edges


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

        super().__init__(bottom_face, top_face)

        # there are 4 side edges: rotate each vertex of bottom_face
        # by angle/2
        side_points = [f.arbitrary_rotation(p, self.axis, self.angle / 2, self.origin) for p in self.base.points]
        for i, point in enumerate(side_points):
            self.block.add_edge(i, i+4, point, kind='arc')

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
