"""Contains all data that user can specify about a Block."""
from typing import List, Dict

from classy_blocks.types import PointListType, OrientType, EdgeKindType

from classy_blocks.data.side import SideData
from classy_blocks.data.edge import EdgeData

from classy_blocks.util import constants as c

class BlockData:
    """User-provided data for a block"""
    def __init__(self, points: PointListType):
        # a list of 8 Vertex objects for each corner of the block
        self.points = points

        # a list of *args for Edges
        self.edges:List[EdgeData] = []

        # Side objects define patch names and projections
        self.sides:Dict[OrientType, SideData] = {o:SideData(o) for o in c.FACE_MAP}

        # block grading;
        # when adding blocks, store chop() parameters;
        # BlockOps will use them to create Gradings()
        self.chops = [[], [], []]

        # cellZone to which the block belongs to
        self.cell_zone = ""

        # written as a comment after block definition
        # (visible in blockMeshDict, useful for debugging)
        self.description = ""

    def add_edge(self, index_1:int, index_2:int,
        kind:EdgeKindType, *args):
        """Adds an edge between vertices at specified indexes.

        Args:
            index_1, index_2: local Block/Face indexes of vertices between which the edge is placed
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
        # here in Block, Curve objects are created; they are converted to Edges later

        # add points for definition
        self.edges.append(EdgeData(index_1, index_2, kind, list(args)))

    def get_edge(self, index_1:int, index_2:int) -> EdgeData:
        """Returns an existing edge if it's defined
        between points at index_1 and index_2
        or raises an exception if it doesn't exist"""
        for edge in self.edges:
            if {edge.index_1, edge.index_2} == {index_1, index_2}:
                return edge
            
        raise RuntimeError(f"Edge not found: {index_1}, {index_2}")

    # def set_patch(self, orients: Union[OrientType, List[OrientType]], patch_name: str) -> None:
    #     """assign one or more block faces (constants.FACE_MAP)
    #     to a chosen patch name"""
    #     # see patches: an example in __init__()

    #     if isinstance(orients, str):
    #         orients = [orients]

    #     for orient in orients:
    #         if self.sides[orient].patch is not None:
    #             warnings.warn(f"Replacing patch {self.sides[orient].patch} with {patch_name}")

    #         self.sides[orient].patch = patch_name

    def chop(self, axis: int, **kwargs: float) -> None:
        """Set block's cell count/size and grading for a given direction/axis.
        Exactly two of the following keyword arguments must be provided:

        :Keyword Arguments:
        * *start_size:
            size of the first cell (last if invert==True)
        * *end_size:
            size of the last cell
        * *c2c_expansion:
            cell-to-cell expansion ratio
        * *count:
            number of cells
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
            https://cfd.direct/openfoam/user-guide/v9-blockMesh/#multi-grading
        """
        self.chops[axis].append(kwargs)

    # def project_face(self, orient:OrientType, geometry: str, edges: bool = False) -> None:
    #     """Assign one or more block faces (self.face_map)
    #     to be projected to a geometry (defined in Mesh)"""
    #     assert orient in c.FACE_MAP

    #     self.sides[orient].project = geometry

    #     if edges:
    #         for i in range(4):
    #             self.add_edge(i, (i + 1) % 4, 'project', geometry)

    @property
    def data(self) -> List['BlockData']:
        return [self]
