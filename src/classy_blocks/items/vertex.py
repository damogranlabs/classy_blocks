"""Defines a numbered vertex in 3D space and all operations
that can be applied to it."""
from typing import Union

from classy_blocks.types import PointType

from classy_blocks.data.point import Point
from classy_blocks.util.constants import vector_format

class Vertex(Point):
    """A 3D point in space with all transformations and an assigned index"""
    def __init__(self, position:Union[PointType, Point], index:int):
        if not isinstance(position, Point):
            position = Point(position)

        super().__init__(position.pos)

        self.index = index

        # TODO: project

    @property
    def description(self) -> str:
        """ Returns a string representation to be written to blockMeshDict"""
        return f"{vector_format(self.pos)} // {self.index}"

    def __eq__(self, other):
        return self.index == other.index

    def __repr__(self):
        return f"Vertex {self.index}"
