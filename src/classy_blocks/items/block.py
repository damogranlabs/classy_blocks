import warnings
from typing import List, Literal, Union, Dict

from classy_blocks.types import PointListType, AxisType, OrientType, EdgeKindType

from classy_blocks.construct.edges import Project
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.edges.factory import factory
from classy_blocks.items.wire import Wire
from classy_blocks.items.axis import Axis

from classy_blocks.items.side import Side
from classy_blocks.grading.chop import Chop

from classy_blocks.util import constants

class Block:
    """A Block and everything that belongs to it"""
    def __init__(self, index:int, vertices:List[Vertex]):
        # index in blockMeshDict
        self.index = index

        # vertices, edges, counts and gradings
        self.vertices = vertices

        # Storing and retrieving pairs of vertices a.k.a. 'wires';
        # self.wires can be indexed so that the desired Wire can be accessed directly;
        # for instance, an edge between vertices 2 and 6 is obtained with
        # self.wires[2][6].edge
        self.wires:List[Dict[int, Wire]] = [{} for _ in range(8)]
        # wires of each axis
        axis_wires = [[], [], []]

        # create wires and connections for quicker addressing
        for axis in range(3):
            for pair in constants.AXIS_PAIRS[axis]:
                wire = Wire(self.vertices, axis, pair[0], pair[1])

                self.wires[pair[0]][pair[1]] = wire
                self.wires[pair[1]][pair[0]] = wire

                axis_wires[axis].append(wire)

        self.axes = [Axis(i, axis_wires[i]) for i in (0, 1, 2)]

        # Side objects define patch names and projections
        self.sides = {o:Side(self.vertices, o) for o in constants.FACE_MAP}

        # cellZone to which the block belongs to
        self.cell_zone = ""

        # written as a comment after block definition
        # (visible in blockMeshDict, useful for debugging)
        self.comment = ""

    def add_edge(self, corner_1:int, corner_2:int, edge:Edge):
        """Adds an edge between vertices at specified indexes."""
        assert 0 <= corner_1 < 8 and 0 <= corner_2 < 8, "Use block-local indexing (0...7)"
        self.wires[corner_1][corner_2].edge = edge

    def chop(self, axis: AxisType, chop:Chop) -> None:
        """Set block's cell count/size and grading for a given direction/axis.
        Exactly two of the following keyword arguments must be provided:

        :Keyword Arguments (see Chop object):
        * *count:
            number of cells;
            Optionally, this can be the only provided argument;
            in that case c2c_expansion will be set to 1.
        * *start_size:
            size of the first cell (last if invert==True)
        * *end_size:
            size of the last cell
        * *c2c_expansion:
            cell-to-cell expansion ratio
        * *total_expansion:
            ratio between first and last cell size

        :Optional keyword arguments:
        * *invert:
            reverses grading if True
        * *take:
            must be 'min', 'max', or 'avg'; takes minimum or maximum edge
            length for block size calculation, or average of all edges in given direction.
            With multigrading only the first 'take' argument is used, others are copied.
        * *length_ratio:
            in case the block is graded using multiple gradings, specify
            length of current division; see
            https://cfd.direct/openfoam/user-guide/v9-blockMesh/#multi-grading;
            Multiple gradings are specified by multiple calls to .chop() with
            the same 'axis' parameter."""
        self.axes[axis].chop(chop)

    def get_axis_wires(self, axis:AxisType) -> List[Wire]:
        """Returns a list of wires that run in the given axis"""
        return self.axes[axis].wires

    def add_neighbour(self, candidate:'Block') -> None:
        """Add a block to neighbours, if applicable"""
        if candidate == self:
            return

        # axes
        for this_axis in self.axes:
            for cnd_axis in candidate.axes:
                this_axis.add_neighbour(cnd_axis)

        # wires
        for this_wire in self.wire_list:
            for cnd_wire in candidate.wire_list:
                this_wire.add_coincident(cnd_wire)

    @property
    def wire_list(self) -> List[Wire]:
        """A flat list of all wires"""
        return self.axes[0].wires + self.axes[1].wires + self.axes[2].wires

    @property
    def edge_list(self) -> List[Edge]:
        """A list of edges from all wires"""
        all_edges = []

        for wire in self.wire_list:
            if wire.edge.kind != 'line':
                all_edges.append(wire.edge)

        return all_edges

    def set_patch(self,
                  orients: Union[OrientType, List[OrientType]],
                  patch_name: str,
                  patch_type:str='patch') -> None:
        """assign one or more block sides (constants.FACE_MAP) to a chosen patch name;
        if type is not specified, it will becom 'patch'"""
        if isinstance(orients, str):
            orients = [orients]

        for orient in orients:
            if self.sides[orient].patch_name is not None:
                warnings.warn(f"Replacing patch {self.sides[orient].patch_name} with {patch_name}")

            self.sides[orient].patch_name = patch_name
            self.sides[orient].patch_type = patch_type

    def project_face(self,
                     orient:OrientType,
                     geometry:Union[List[str], str],
                     edges:bool=True) -> None:
        """Assign one or more block faces (self.face_map)
        to be projected to a geometry (defined in Mesh)"""
        assert orient in constants.FACE_MAP

        if isinstance(geometry, str):
            geometry = [geometry]

        self.sides[orient].project_to = geometry

        if edges:
            corners = self.sides[orient].corners

            for i, corner_1 in enumerate(corners):
                corner_2 = corners[(i+1) % 4]

                edge = factory.create(
                    self.vertices[corner_1],
                    self.vertices[corner_2],
                    Project(geometry)
                )
                self.add_edge(corner_1, corner_2, edge)

    @property
    def is_defined(self) -> bool:
        """Returns True if counts and gradings are defined for all axes"""
        # TODO: TEST
        return all(axis.is_defined for axis in self.axes)

    def copy_grading(self) -> bool:
        """Attempts to copy grading from a neighbouring block;
        Returns True if the block is/has been defined, False
        if the block still has missing data"""
        # TODO: TEST
        if self.is_defined:
            return False
        
        updated = False

        if not self.is_defined:
            for axis in self.axes:
                updated = axis.copy_grading() or updated

        return updated

    @property
    def description(self) -> str:
        """hex definition for blockMesh"""
        # TODO: test
        out = "\thex "

        # vertices
        out += " ( " + " ".join(str(v.index) for v in self.vertices) + " ) "

        # cellZone
        out += self.cell_zone

        # number of cells
        out += " (" + " ".join([str(axis.grading.count) for axis in self.axes]) + " ) "

        # grading
        # TODO: no daisy.object.chaining
        out += " simpleGrading (" + \
            self.axes[0].grading.description + " " + \
            self.axes[1].grading.description + " " + \
            self.axes[2].grading.description + ") "

        # add a comment with block index
        out += f" // {self.index} {self.comment}\n"

        return out
