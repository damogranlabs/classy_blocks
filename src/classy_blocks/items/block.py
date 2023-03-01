import warnings
from typing import List, Literal, Union

from classy_blocks.types import PointListType, AxisType, OrientType, EdgeKindType

from classy_blocks.data.block_data import BlockData
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.wireframe import Wireframe

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
        self.edges:List[Edge] = []

        self.frame = Wireframe(self.vertices, self.edges)

        # Side objects define patch names and projections
        #self.sides = {o:Side(o) for o in constants.FACE_MAP}

        # cellZone to which the block belongs to
        self.cell_zone = ""

        # written as a comment after block definition
        # (visible in blockMeshDict, useful for debugging)
        self.comment = ""

    def add_edge(self, corner_1:int, corner_2:int, kind:EdgeKindType, *args):
        """Adds an edge between vertices at specified indexes.
        Args:
            corner_1, corner_2: local Block/Face indexes of vertices between which the edge is placed
            kind: edge type that will be written to blockMeshDict.
            *args: provide the following information for edge creation, depending on specified 'kind':
                - Classic OpenFOAM arc definition: kind, arc_point;
                    ..., 'arc', <types.PointType>
                - Origin arc definition (ESI-CFD version*): kind, origin, flatness (optional, default 1)
                    ..., 'origin', <types.PointType>, flatness
                - Angle-and-axis (Foundation version**):
                    ..., kind='angle', angle=<float (in radians)>, axis=<types.VectorType>
                - Spline:
                    ..., kind='spline', points=<types.PointListType>
                - PolyLine:
                    ..., kind='polyLine', points=<types.PointListType>
                - Projected edges (provide geometry with mesh.add_geometry()):
                    ..., kind='project', geometry=str
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
                block.add_edge(0, 1, 'arc', [0.5, 0.25, 0])
            A spline edge with single or multiple points:
                block.add_edge(0, 1, 'spline', [[0.3, 0.25, 0], [0.6, 0.1, 0], [0.3, 0.25, 0]])
            Same points as above but specified as polyLine:
                block.add_edge(0, 1, 'polyLine', [[0.3, 0.25, 0], [0.6, 0.1, 0], [0.3, 0.25, 0]])
            An edge, projected to geometry defined as 'terrain':
                block.add_edge(0, 1, 'project', 'terrain')
            An arc, defined using ESI-CFD's 'origin' style:
                block.add_edge(0, 1, 'origin', [0.5, -0.5, 0], 2)
            An arc, defined using OF Foundation's 'angle and axis' style:
                block.add_edge(0, 1, 'angle', np.pi/6, [0, 0, 1])"""
        assert 0 <= corner_1 < 8 and 0 <= corner_2 < 8, "Use block-local indexing (0...7)"

        #self.edges.append('edge')

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
                # set gradings according to data.axis_chops
        #self.frame.chop_axis(axis, chop)


    # def add_neighbour(self, candidate:'Block') -> None:
    #     """Add a block to neighbours, if applicable"""
    #     if candidate == self:
    #         return

    #     self.frame.add_neighbour(candidate.frame)

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
        out += self.data.cell_zone

        # number of cells
        out += " (" + " ".join([str(axis.grading.count) for axis in self.frame.axes]) + " ) "

        # grading
        out += " simpleGrading (" + \
            self.frame.axes[0].grading.description + " " + \
            self.frame.axes[1].grading.description + " " + \
            self.frame.axes[2].grading.description + ") "

        # add a comment with block index
        out += f" // {self.index} {self.data.comment}\n"

        return out