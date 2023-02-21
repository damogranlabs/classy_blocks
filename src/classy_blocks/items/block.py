from typing import List, Literal

from classy_blocks.data.block_data import BlockData
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.wireframe import Wireframe

from classy_blocks.types import AxisType

class Block:
    """Further operations on blocks"""
    def __init__(self, data:BlockData, index:int, vertices:List[Vertex], edges:List[Edge]):
        self.data = data # user-supplied data
        self.index = index # index in blockMeshDict

        # contains vertices, edges, counts and gradings, and tools for manipulation
        self.vertices = vertices
        self.frame = Wireframe(vertices, edges)

        # set gradings according to data.axis_chops
        for axis in (0, 1, 2):
            self.frame.chop_axis(axis, self.data.axis_chops[axis])

        # set gradings of specific edges
        #for chop in self.data.

    # def _generate_faces(self) -> Dict[OrientType, Face]:
    #     """Generate Face objects from data.sides"""
    #     return {
    #         orient:Face.from_side(self.data.sides[orient], self.vertices)
    #         for orient in c.FACE_MAP
    #     }

    def add_neighbour(self, candidate:'Block') -> None:
        """Add a block to neighbours, if applicable"""
        # TODO: test
        if candidate == self:
            return

        for this_wire in self.frame.wires:
            for cnd_wire in candidate.frame.wires:
                this_wire.add_coincident(cnd_wire)
                cnd_wire.add_coincident(this_wire)

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
        out += (f" ({self.frame.axes[0][0].grading.count}"
            f" {self.frame.axes[1][0].grading.count}"
            f" {self.frame.axes[2][0].grading.count})")

        # add a comment with block index
        out += f" // {self.index} {self.data.comment}\n"

        return out