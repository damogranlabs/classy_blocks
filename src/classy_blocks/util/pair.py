from typing import List

from classy_blocks.items.vertex import Vertex
from classy_blocks.util import constants

class Pair:
    """Represents two vertices that define an edge;
    supplies tools to create and compare, etc"""
    def __init__(self, vertices:List[Vertex], axis:int, index:int):
        assert len(vertices) == 8, "Provide 8 vertices that define a block"
        assert 0 <= axis < 3, "Axis must be between 0 and 3 (x, y, z)"
        assert 0 <= index < 4, "Index must be between 0 and 4 (side corners)"

        self.axis = axis
        self.index = index

        index_1 = constants.AXIS_PAIRS[axis][index][0]
        index_2 = constants.AXIS_PAIRS[axis][index][1]

        self.vertex_1 = vertices[index_1]
        self.vertex_2 = vertices[index_2]

    @property
    def is_valid(self) -> bool:
        """A pair with two equal vertices is useless"""
        return self.vertex_1 != self.vertex_2

    def is_aligned(self, pair:'Pair') -> bool:
        """Returns true is this pair has the same alignment
        as the pair in the argument"""
        if self != pair:
            raise RuntimeError(f"Pairs are not coincident: {self}, {pair}")
        
        return self.vertex_1 == pair.vertex_1

    def __eq__(self, obj):
        # If vertices are the same the pair is the same, regardless of alignment
        return (self.vertex_1 == obj.vertex_1) and (self.vertex_2 == obj.vertex_2) or \
            (self.vertex_1 == obj.vertex_2) and (self.vertex_2 == obj.vertex_1)

    def __str__(self):
        return f"Pair #{self.index} on axis {self.axis}"
