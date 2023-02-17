import dataclasses

from typing import List, Dict, Set, Literal, Tuple

from classy_blocks.types import OrientType

from classy_blocks.data.block import BlockData
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.face import Face
from classy_blocks.items.wireframe import Wireframe


from classy_blocks.grading import Grading
from classy_blocks.util import constants as c
from classy_blocks.util import functions as f

@dataclasses.dataclass
class Neighbour:
    """Stores reference to a neighbour block (the 'original' one
    is the holder of this object) and provides
    means of copying counts and gradings"""
    block:'Block'
    wire:'Wire'

class Block:
    """Further operations on blocks"""
    def __init__(self, data:BlockData, index:int, vertices:List[Vertex], edges:List[Edge]):
        self.data = data # user-supplied data
        self.index = index # index in blockMeshDict

        # contains vertices, edges, counts and gradings, and tools for manipulation
        self.vertices = vertices
        self.frame = Wireframe(vertices, edges)

        # # create Face and Grading objects
        # self.faces = self._generate_faces()
        # self.gradings = self._generate_gradings()

        # # Neighbours are specified for each axis for 
        # # easier and quicker grading/count propagation
        self.neighbours:Set[Neighbour] = set()

    # def _generate_faces(self) -> Dict[OrientType, Face]:
    #     """Generate Face objects from data.sides"""
    #     return {
    #         orient:Face.from_side(self.data.sides[orient], self.vertices)
    #         for orient in c.FACE_MAP
    #     }

    # def _generate_gradings(self) -> List[Grading]:
    #     """Generates Grading() objects from data.chops"""
    #     gradings = [Grading(), Grading(), Grading()]

    #     for i in range(3):
    #         grading = gradings[i]
    #         params = self.data.chops[i]

    #         if len(params) < 1:
    #             # leave the grading empty
    #             continue

    #         block_size = self.get_size(i, take=params[0].pop("take", "avg"))
    #         grading.set_block_size(block_size)

    #         for p in params:
    #             grading.add_division(**p)
        
    #     return gradings

    def add_neighbour(self, candidate:'Block') -> None:
        """Add a block to neighbours, if applicable"""
        # TODO: test
        if candidate == self:
            return

        for this_wire in self.frame.wires:
            for cnd_wire in candidate.frame.wires:
                if this_wire == cnd_wire:
                    # that's the neighbour
                    self.neighbours.add(Neighbour(candidate, this_wire))

    def get_size(self, axis: int, take: Literal["min", "max", "avg"] = "avg") -> float:
        """Returns block dimensions in given axis"""
        edge_lengths = [w.edge.length for w in self.frame.get_axis_wires(axis)]

        if take == "avg":
            return sum(edge_lengths) / len(edge_lengths)

        if take == "min":
            return min(edge_lengths)

        if take == "max":
            return max(edge_lengths)

        raise ValueError(f"Unknown sizing specification: {take}. Available: min, max, avg")

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