from typing import List

from classy_blocks.items.vertex import Vertex
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.edges.line import LineEdge

class Wire:
    """Represents two vertices that define an edge;
    supplies tools to create and compare, etc"""
    def __init__(self, vertices:List[Vertex], axis:int, corner_1:int, corner_2:int):
        self.corner_1 = corner_1
        self.corner_2 = corner_2

        self.vertex_1 = vertices[corner_1]
        self.vertex_2 = vertices[corner_2]

        self.axis = axis

        # the default edge is 'line' but will be replaced if the user wishes so
        self.edge:Edge = LineEdge(self.vertex_1, self.vertex_2)

        # a wire can have up to 3 neighbours but is added with self.add_neighbour()
        self.neighbours:List['Wire'] = []

    @property
    def is_valid(self) -> bool:
        """A pair with two equal vertices is useless"""
        return self.vertex_1 != self.vertex_2

    def is_aligned(self, pair:'Wire') -> bool:
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
        return f"Wire<{self.corner_1}-{self.corner_2}>"
