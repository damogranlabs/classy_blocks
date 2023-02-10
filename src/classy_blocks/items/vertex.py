"""Defines a numbered vertex in 3D space and all operations
that can be applied to it."""
from classy_blocks.data.point import Point
from classy_blocks.util.constants import vector_format

class Vertex(Point):
    """A 3D point in space with all transformations and an assigned index"""
    def __init__(self, position:Point, index:int):
        super().__init__(position)
        self.index = index

    def output(self) -> str:
        """ Returns a string representation to be written to blockMeshDict"""
        return f"\t{vector_format(self.pos)} // {self.index}\n"
