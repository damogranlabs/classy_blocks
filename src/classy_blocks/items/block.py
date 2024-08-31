from typing import List, get_args

from classy_blocks.grading.chop import Chop
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.wires.axis import Axis
from classy_blocks.items.wires.wire import Wire
from classy_blocks.types import AxisType, IndexType
from classy_blocks.util import constants
from classy_blocks.util.frame import Frame


class Block:
    """A Block and everything that belongs to it"""

    def __init__(self, index: int, vertices: List[Vertex]):
        # index in blockMeshDict
        self.index = index

        # vertices, edges, counts and gradings
        self.vertices = vertices

        # wires and axes
        self.wires = Frame[Wire]()

        # create wires and connections for quicker addressing
        for axis in range(3):
            for pair in constants.AXIS_PAIRS[axis]:
                wire = Wire(self.vertices, axis, pair[0], pair[1])
                self.wires.add_beam(pair[0], pair[1], wire)

        self.axes = [Axis(i, self.wires.get_axis_beams(i)) for i in get_args(AxisType)]

        # cellZone to which the block belongs to
        self.cell_zone: str = ""

        # written as a comment after block definition
        # (visible in blockMeshDict, useful for debugging)
        self.comment: str = ""

    def add_edge(self, corner_1: int, corner_2: int, edge: Edge):
        """Adds an edge between vertices at specified indexes."""
        if not (0 <= corner_1 < 8 and 0 <= corner_2 < 8):
            raise ValueError(
                f"Invalid corner 1 ({corner_1}) or corner 2 ({corner_2}) index. Use block-local indexing (0...7)."
            )

        self.wires[corner_1][corner_2].edge = edge

    def chop(self, axis: AxisType, chop: Chop) -> None:
        """Set block's cell count/size and grading for a given direction/axis.
        Exactly two of the following keyword arguments must be provided:

        :Keyword Arguments (see Chop object):
        * *count:
            number of cells;
            Optionally, this can be the only provided argument;
            in that case c2c_expansion will be set to 1.
        * *start_size:
            size of the first cell
        * *end_size:
            size of the last cell
        * *c2c_expansion:
            cell-to-cell expansion ratio
        * *total_expansion:
            ratio between first and last cell size

        :Optional keyword arguments:
        * *preserve:
            which of the specified values should be preserved. Must be one of
            "start_size", "end_size" or "c2c_expansion". The last is default and will produce
            regular simpleGrading with 3 values for each axis. When start or end size
            is to be kept, grading will switch to edgeGrading so that cells on each edge
            will stay consistent start/end size regardless of edge length.
        * *length_ratio:
            in case the block is graded using multiple gradings, specify
            length of current division; see
            https://cfd.direct/openfoam/user-guide/v9-blockMesh/#multi-grading;
            Multiple gradings are specified by multiple calls to .chop() with
            the same 'axis' parameter."""
        self.axes[axis].chop(chop)

    def get_axis_wires(self, axis: AxisType) -> List[Wire]:
        """Returns a list of wires that run in the given axis"""
        return self.wires.get_axis_beams(axis)

    def add_neighbour(self, candidate: "Block") -> None:
        """Add a block to neighbours, if applicable"""
        if candidate == self:
            return

        # axes
        for this_axis in self.axes:
            for cnd_axis in candidate.axes:
                this_axis.add_neighbour(cnd_axis)
                this_axis.add_sequential(cnd_axis)

        # wires
        for this_wire in self.wire_list:
            for cnd_wire in candidate.wire_list:
                this_wire.add_coincident(cnd_wire)

    @property
    def wire_list(self) -> List[Wire]:
        """A flat list of all wires"""
        # TODO: no daisy chaining!
        return self.axes[0].wires.wires + self.axes[1].wires.wires + self.axes[2].wires.wires

    @property
    def edge_list(self) -> List[Edge]:
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

    def grade(self):
        for axis in self.axes:
            axis.grade()

    def copy_grading(self) -> bool:
        """Attempts to copy grading from a neighbouring block;
        returns True if the grading was copied and False in all other cases"""
        updated = False

        if not self.is_defined:
            for axis in self.axes:
                updated = axis.copy_grading() or updated

        return updated

    def check_consistency(self) -> None:
        for axis in self.axes:
            axis.check_consistency()

    @property
    def indexes(self) -> IndexType:
        return [vertex.index for vertex in self.vertices]

    def format_grading(self) -> str:
        """Returns the simple/edgeGrading string"""
        if all(axis.is_simple for axis in self.axes):  # is_simple
            return (
                "simpleGrading ( "
                + self.axes[0].wires.format_single()
                + " "
                + self.axes[1].wires.format_single()
                + " "
                + self.axes[2].wires.format_single()
                + " )"
            )
        else:
            return (
                "edgeGrading ( "
                + self.axes[0].wires.format_all()
                + " "
                + self.axes[1].wires.format_all()
                + " "
                + self.axes[2].wires.format_all()
                + " )"
            )

    @property
    def description(self) -> str:
        """hex definition for blockMesh"""
        fmt_vertices = "( " + " ".join(str(v.index) for v in self.vertices) + " )"
        fmt_count = "( " + " ".join([str(axis.count) for axis in self.axes]) + " )"

        fmt_grading = self.format_grading()
        fmt_comments = f"// {self.index} {self.comment}\n"

        return f"\thex {fmt_vertices} {self.cell_zone} {fmt_count} {fmt_grading} {fmt_comments}"
