import warnings
from typing import List, Literal, Union, Dict

from classy_blocks.types import PointListType, AxisType, OrientType, EdgeKindType

from classy_blocks.data.edges import EdgeData
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
    def __init__(self, points:PointListType):
        # index in blockMeshDict; a proper value
        # will be assigned when the this block is added to mesh
        self.index = -1

        # vertices, edges, counts and gradings
        self.vertices = [Vertex(point) for point in points]
        
        # Storing and retrieving pairs of vertices a.k.a. 'wires';
        # Block object can be indexed so that the desired Wire can be accessed directly;
        # for instance, an edge between vertices 2 and 6 is obtained with
        # self.wires[2][6].edge"""

        # the opposite side of each vertex
        self.wires:List[Dict[int, Wire]] = [{} for _ in range(8)]
        # wires of each axis
        self.axes = [Axis(i) for i in (0, 1, 2)]

        # create wires and connections for quicker addressing
        for axis in range(3):
            for pair in constants.AXIS_PAIRS[axis]:
                wire = Wire(self.vertices, axis, pair[0], pair[1])

                self.wires[pair[0]][pair[1]] = wire
                self.wires[pair[1]][pair[0]] = wire

                self.axes[axis].wires.append(wire)

        # Side objects define patch names and projections
        self.sides = None #{o:Side(o) for o in constants.FACE_MAP}

        # cellZone to which the block belongs to
        self.cell_zone = ""

        # written as a comment after block definition
        # (visible in blockMeshDict, useful for debugging)
        self.comment = ""

    def add_edge(self, corner_1:int, corner_2:int, data:EdgeData):
        """Adds an edge between vertices at specified indexes.
        Args:
            corner_1, corner_2: local Block/Face indexes of vertices between which the edge is placed
            data: an EdgeData object from cb.edges.* containing required info for edge creation:
            - Arc: classic OpenFOAM arc definition with arc_point
            - Origin: ESI-CFD version* of arc with origin point and optional flatness (default 1)
            - Angle: Foundation version** with sector angle and axis
            - Spline: spline passing through a list of points
            - PolyLine: a series of lines between a list of points
            - Project: project to one surface or to an intersection of two

        Definition of arc edges:
            * ESI-CFD version
            https://www.openfoam.com/news/main-news/openfoam-v20-12/pre-processing#x3-22000
            https://develop.openfoam.com/Development/openfoam/-/blob/master/src/mesh/blockMesh/blockEdges/arcEdge/arcEdge.H
            ** Foundation version:
            https://github.com/OpenFOAM/OpenFOAM-10/commit/73d253c34b3e184802efb316f996f244cc795ec6
            All arc variants are supported by classy_blocks;
            however, only the first one will be written to blockMeshDict for compatibility.
            If an edge was specified by #2 or #3, the definition will be output as a comment next
            to that edge definition.
        Examples:
            Add an arc edge:
                block.add_edge(0, 1, Arc([0.5, 0.25, 0]))
            A spline edge with single or multiple points:
                block.add_edge(0, 1, Spline([[0.3, 0.25, 0], [0.6, 0.1, 0], [0.3, 0.25, 0]]))
            Same points as above but specified as polyLine:
                block.add_edge(0, 1, PolyLine([[0.3, 0.25, 0], [0.6, 0.1, 0], [0.3, 0.25, 0]]))
            An edge, projected to geometry defined as 'terrain':
                block.add_edge(0, 1, Project('terrain'))
            An edge, projected to intersection of 'terrain' and 'wall'
                block.add_edge(0, 1, Project(['terrain', 'wall']))
            An arc, defined using ESI-CFD's 'origin' style:
                block.add_edge(0, 1, Origin([0.5, -0.5, 0]))
            An arc, defined using OF Foundation's 'angle and axis' style:
                block.add_edge(0, 1, Angle(np.pi/6, [0, 0, 1]))"""
        assert 0 <= corner_1 < 8 and 0 <= corner_2 < 8, "Use block-local indexing (0...7)"

        edge = factory.create(self.vertices[corner_1], self.vertices[corner_2], data)
        self.wires[corner_1][corner_2].edge = edge

    def chop(self, axis: AxisType, **kwargs:Union[str, float, int, bool]) -> None:
        """Set block's cell count/size and grading for a given direction/axis.
        Exactly two of the following keyword arguments must be provided:

        :Keyword Arguments:
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
        self.axes[axis].chop(Chop(**kwargs))

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
        patch_type:str='patch'
    ) -> None:
        """assign one or more block sides (constants.FACE_MAP) to a chosen patch name;
        if type is not specified, it will becom 'patch'"""
        # if isinstance(orients, str):
        #     orients = [orients]

        # for orient in orients:
        #     if self.sides[orient].patch_name is not None:
        #         warnings.warn(f"Replacing patch {self.sides[orient].patch_name} with {patch_name}")

        #     self.sides[orient].patch_name = patch_name
        #     self.sides[orient].patch_type = patch_type

    # def project_face(self, orient:OrientType, geometry: str, edges: bool = False) -> None:
    #     """Assign one or more block faces (self.face_map)
    #     to be projected to a geometry (defined in Mesh)"""
    #     assert orient in constants.FACE_MAP

    #     self.sides[orient].project = geometry

    #     if edges:
    #         for i in range(4):
    #             self.add_edge(i, (i + 1) % 4, 'project', geometry)

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
        out += " simpleGrading (" + \
            self.axes[0].grading.description + " " + \
            self.axes[1].grading.description + " " + \
            self.axes[2].grading.description + ") "

        # add a comment with block index
        out += f" // {self.index} {self.comment}\n"

        return out