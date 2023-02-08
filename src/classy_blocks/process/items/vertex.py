"""Defines a numbered vertex in 3D space and all operations
that can be applied to it."""
from classy_blocks.process.items.point import Point
from classy_blocks.util.constants import vector_format


class Vertex:
    """A 3D point in space with all transformations and an assigned index"""
    def __init__(self, point:Point, index:int):
        self.point = point
        self.index = index

    def output(self) -> str:
        """ Returns a string representation to be written to blockMeshDict"""
        return f"\t{vector_format(self.point.position)} // {self.index}\n"
