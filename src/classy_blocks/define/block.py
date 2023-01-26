"""Contains all data to place a block into mesh."""
from typing import List, Union, Dict

import warnings

from classy_blocks.types import OrientType, EdgeKindType

from classy_blocks.define.point import Point
from classy_blocks.define.side import Side
from classy_blocks.define import edge

from classy_blocks.util import constants as c

class Block:
    """a direct representation of a blockMesh block;
    contains all necessary data to create it."""
    def __init__(self, points: List[Point]):
        # a list of 8 Vertex and Edge objects for each corner/edge of the block
        self.points = points
        self.edges: List[edge.Edge] = []

        # generate Side objects:
        self.sides:Dict[OrientType, Side] = {o:Side(o) for o in c.FACE_MAP}

        # block grading;
        # when adding blocks, store chop() parameters;
        # use them in mesh.write()
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
        args = [
            self.points[index_1],
            self.points[index_2],
            kind
        ] + list(args)

        self.edges.append(edge.factory.create(*args))

    def set_patch(self, orients: Union[OrientType, List[OrientType]], patch_name: str) -> None:
        """assign one or more block faces (constants.FACE_MAP)
        to a chosen patch name"""
        # see patches: an example in __init__()

        if isinstance(orients, str):
            orients = [orients]

        for orient in orients:
            if self.sides[orient].patch is not None:
                warnings.warn(f"Replacing patch {self.sides[orient].patch} with {patch_name}")

            self.sides[orient].patch = patch_name

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
        # Actual Grading will be calculated within mesh.prepare() > process.lists.blocks
        self.chops[axis].append(kwargs)

    def project_face(self, orient:OrientType, geometry: str, edges: bool = False) -> None:
        """Assign one or more block faces (self.face_map)
        to be projected to a geometry (defined in Mesh)"""
        assert orient in c.FACE_MAP

        self.sides[orient].project = geometry

        if edges:
            for i in range(4):
                self.add_edge(i, (i + 1) % 4, 'project', geometry)
