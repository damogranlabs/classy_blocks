from typing import get_args

from classy_blocks.cbtyping import DirectionType, IndexType, OrientType
from classy_blocks.grading.chop import Chop
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.wires.axis import Axis
from classy_blocks.items.wires.wire import Wire
from classy_blocks.util import constants
from classy_blocks.util.frame import Frame


class Block:
    """A Block and everything that belongs to it"""

    def __init__(self, index: int, vertices: list[Vertex]):
        # index in blockMeshDict
        self.index = index

        # vertices, edges, counts and gradings
        self.vertices = vertices

        # wires and axes
        self.wires = Frame[Wire]()

        # create wires and connections for quicker addressing
        for direction in get_args(DirectionType):
            for pair in constants.AXIS_PAIRS[direction]:
                wire = Wire(self.vertices, direction, pair[0], pair[1])
                self.wires.add_beam(pair[0], pair[1], wire)

        self.axes = [Axis(i, self.wires.get_axis_beams(i)) for i in get_args(DirectionType)]

        # cellZone to which the block belongs to
        self.cell_zone: str = ""

        # written as a comment after block definition
        # (visible in blockMeshDict, useful for debugging)
        self.comment: str = ""

        # 'hidden' blocks carry all of the data from the mesh
        # but is not inserted into blockMeshDict;
        self.visible = True

    def add_edge(self, corner_1: int, corner_2: int, edge: Edge):
        """Adds an edge between vertices at specified indexes."""
        if not (0 <= corner_1 < 8 and 0 <= corner_2 < 8):
            raise ValueError(
                f"Invalid corner 1 ({corner_1}) or corner 2 ({corner_2}) index. Use block-local indexing (0...7)."
            )

        self.wires[corner_1][corner_2].edge = edge

    def get_axis_wires(self, direction: DirectionType) -> list[Wire]:
        """Returns a list of wires that run in the given axis"""
        return self.wires.get_axis_beams(direction)

    def get_axis_direction(self, axis: Axis) -> DirectionType:
        for i in get_args(DirectionType):
            if self.axes[i] == axis:
                return i

        # TODO: use a custom exception
        raise RuntimeError("Axis not in this block!")

    def add_neighbour(self, candidate: "Block") -> None:
        """Add a block to neighbours, if applicable"""
        if candidate == self:
            return

        # axes
        for this_axis in self.axes:
            for cnd_axis in candidate.axes:
                this_axis.add_neighbour(cnd_axis)
                this_axis.add_inline(cnd_axis)

        # wires
        for this_wire in self.wire_list:
            for cnd_wire in candidate.wire_list:
                this_wire.add_coincident(cnd_wire)

    def add_chops(self, direction: DirectionType, chops: list[Chop]) -> None:
        self.axes[direction].chops += chops

    def update_wires(self) -> None:
        for wire in self.wire_list:
            # set actual grading.length after adding edges
            wire.update()

    def grade(self) -> None:
        for axis in self.axes:
            axis.grade()

    @property
    def wire_list(self) -> list[Wire]:
        """A flat list of all wires"""
        # TODO: no daisy chaining!
        return self.axes[0].wires.wires + self.axes[1].wires.wires + self.axes[2].wires.wires

    @property
    def edge_list(self) -> list[Edge]:
        """A list of edges from all wires"""
        all_edges = []

        for wire in self.wire_list:
            if wire.edge.kind != "line":
                all_edges.append(wire.edge)

        return all_edges

    @property
    def is_defined(self) -> bool:
        """Returns True if counts and gradings are defined for all axes"""
        return all(axis.is_defined for axis in self.axes)

    def check_consistency(self) -> None:
        for axis in self.axes:
            axis.check_consistency()

    @property
    def indexes(self) -> IndexType:
        return [vertex.index for vertex in self.vertices]

    def get_side_vertices(self, orient: OrientType) -> list[Vertex]:
        return [self.vertices[i] for i in constants.FACE_MAP[orient]]

    def __hash__(self) -> int:
        return self.index

    def __repr__(self) -> str:
        return f"Block {self.index}"
