import functools

from typing import List, Dict, Set, Literal, Tuple

from classy_blocks.types import OrientType

from classy_blocks.data.block import BlockData
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.face import Face
from classy_blocks.grading import Grading

from classy_blocks.util.pair import Pair
from classy_blocks.util import constants as c
from classy_blocks.util import functions as f

class Block:
    """Further operations on blocks"""
    def __init__(self, data:BlockData, index:int, vertices:List[Vertex], edges:List[Edge]):
        self.data = data
        self.index = index

        self.vertices = vertices
        self.edges = edges

        # create Face and Grading objects
        self.faces = self._generate_faces()
        self.gradings = self._generate_gradings()

        self.neighbours:Set[Block] = set()

    def _generate_faces(self) -> Dict[OrientType, Face]:
        """Generate Face objects from data.sides"""
        return {
            orient:Face.from_side(self.data.sides[orient], self.vertices)
            for orient in c.FACE_MAP
        }

    def _generate_gradings(self) -> List[Grading]:
        """Generates Grading() objects from data.chops"""
        gradings = [Grading(), Grading(), Grading()]

        for i in range(3):
            grading = gradings[i]
            params = self.data.chops[i]

            if len(params) < 1:
                # leave the grading empty
                continue

            block_size = self.get_size(i, take=params[0].pop("take", "avg"))
            grading.set_block_size(block_size)

            for p in params:
                grading.add_division(**p)
        
        return gradings

    def find_edge(self, index_1: int, index_2: int) -> Edge:
        """Returns edges between given vertex indexes;
        the indexes in parameters refer to internal block numbering"""
        for edge in self.edges:
            if {edge.vertex_1.index, edge.vertex_2.index} == \
                {index_1, index_2}:
                return edge

        raise RuntimeError(f"Edge not found: {index_1} {index_2}")

    def add_neighbour(self, block:'Block') -> None:
        """Add a block to neighbours, if applicable"""
        # TODO: test
        if block == self:
            return

        for pair in self.vertex_pairs:
            if pair in block.vertex_pairs:
                self.neighbours.add(block)

    def get_size(self, axis: int, take: Literal["min", "max", "avg"] = "avg") -> float:
        """Returns block dimensions in given axis"""
        # if an edge is defined, use the edge.get_length(),
        # otherwise simply distance between two points
        def vertex_distance(index_1:int, index_2:int) -> float:
            return f.norm(self.vertices[index_1].pos - self.vertices[index_2].pos)

        def block_size(index_1:int, index_2:int) -> float:
            try:
                return self.find_edge(index_1, index_2).length
            except RuntimeError:
                return vertex_distance(index_1, index_2)

        edge_lengths = [block_size(pair[0], pair[1]) for pair in c.AXIS_PAIRS[axis]]

        if take == "avg":
            return sum(edge_lengths) / len(edge_lengths)

        if take == "min":
            return min(edge_lengths)

        if take == "max":
            return max(edge_lengths)

        raise ValueError(f"Unknown sizing specification: {take}. Available: min, max, avg")

    @functools.cached_property
    def vertex_pairs(self) -> List[Pair]:
        """Returns 12 Pair objects for each 'edge' of each axis"""
        pairs = []

        for axis in range(3):
            for index in range(4):
                pairs.append(Pair(self.vertices, axis, index))

        return pairs

    def get_axis_from_pair(self, other_pair: Pair) -> Tuple[int, bool]:
        """returns axis index and orientation from a given pair of vertices;
        orientation is True if blocks are aligned or false when inverted."""
        for this_pair in self.vertex_pairs:
            if this_pair == other_pair:
                return this_pair.axis, this_pair.is_aligned(other_pair)
        
        raise RuntimeError(f"No such pair of vertices in this block: {other_pair}")

    @property
    def description(self) -> str:
        """hex definition for blockMesh"""
        # TODO: test
        out = "\thex "

        # vertices
        out += " ( " + " ".join(str(v.index) for v in self.vertices) + " ) "

        # cellZone
        out += self.data.cell_zone

        # number of cells
        #grading = self.gradings[i]

        #out += f" ({grading[0].count} {grading[1].count} {grading[2].count}) "
        # grading
        #out += f" ({grading[0].grading} {grading[1].grading} {grading[2].grading})"

        out += ' (10 10 10) simpleGrading (1 1 1) '

        # add a comment with block index
        out += f" // {self.index} {self.data.comment}\n"

        return out

        print(edge_lengths)