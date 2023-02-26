import abc
from typing import List

import numpy as np

from classy_blocks.types import PointType, PointListType, EdgeKindType

from classy_blocks.base.transformable import TransformableBase
from classy_blocks.data.point import Point
from classy_blocks.data.edge_data import EdgeData

class DataObjectBase(abc.ABC, TransformableBase):
    """An abstract class defining common properties for all
    data definition objects"""
    def __init__(self, points:PointListType):
        # a list of 8 Vertex objects for each corner of the block
        self.points = [Point(pos) for pos in points]

        # a list of *args for Edges
        self.edges:List[EdgeData] = []

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

        self.edges.append(EdgeData(corner_1, corner_2, kind, list(args)))

    def get_edge(self, corner_1:int, corner_2:int) -> EdgeData:
        """Returns an existing edge if it's defined
        between points at corner_1 and corner_2
        or raises an exception if it doesn't exist"""
        for edge in self.edges:
            if {edge.corner_1, edge.corner_2} == {corner_1, corner_2}:
                return edge

        raise RuntimeError(f"Edge not found: {corner_1}, {corner_2}, is it defined already?")

    @property
    def center(self) -> PointType:
        """Returns the center point of this face."""
        # TODO: cache?
        return np.average(self.points, axis=0)

    @property
    @abc.abstractmethod
    def blocks(self) -> List['DataObjectBase']:
        """A list of single block (Block/Operation)
        or multiple blocks defining this object; to be added to Mesh"""
