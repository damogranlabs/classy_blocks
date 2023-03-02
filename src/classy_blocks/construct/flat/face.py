from typing import List, Optional

import copy

import numpy as np

from classy_blocks.types import VectorType, PointType, PointListType
from classy_blocks.base.transformable import TransformableBase
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.edges.factory import factory
from classy_blocks.util import constants
from classy_blocks.util import functions as f

class Face(TransformableBase):
    """A collection of 4 Vertices and optionally 4 Edges,
    creating an arbitrary quadrangle.

    Args:
        points: a list or a numpy array of exactly 4 points in 3d space
        edges: an optional list of data for edge creation;
            if provided, it must be have exactly 4 elements,
            each element a list of data for edge creation; the format is
            the same as passed to Block.add_edge(). Each element of the list
            represents an edge between its corner and the next, for instance:

            - edges=[None, ['arc', [0.4, 1, 1]], None, None] will create an
            arc edge between the 1st and the 2nd vertex of this face
            - edges=[['project', 'terrain']*4] will project all 4 edges
            of this face: 0-1, 1-2, 2-3, 3-0.
        
        check_coplanar: if True, a ValueError will be raised given non-coplanar points
    """
    def __init__(self, points:PointListType, edges:Optional[List]=None, check_coplanar:bool=False):
        points = np.asarray(points)
        if np.shape(points) != (4, 3):
            raise ValueError("Provide exactly 4 points in 3D space")

        if check_coplanar:
            if abs(
                np.dot(
                    points[1] - points[0],
                    np.cross(points[3] - points[0], points[2] - points[0])
                )) > constants.tol:
                raise ValueError("Points are not coplanar!")

            # TODO: coplanar edges?

        self.points = points
        self.edges = edges

    def translate(self, displacement: VectorType) -> 'Face':
        for point in self.points:
            vertex.translate(displacement)

        for edge in self.edges:
            edge.translate(displacement)

        return self

    def rotate(self, angle: float, axis: VectorType, origin: Optional[PointType] = None) -> 'Face':
        for vertex in self.vertices:
            vertex.rotate(angle, axis, origin)
        
        for edge in self.edges:
            edge.rotate(angle, axis, origin)

        return self
    
    def scale(self, ratio: float, origin: Optional[PointType] = None) -> 'Face':
        for vertex in self.vertices:
            vertex.scale(ratio, origin)
        
        for edge in self.edges:
            edge.scale(ratio, origin)
        
        return self

    def invert(self):
        """Reverses the order of points in this face."""
        self.vertices.reverse()
    
    def copy(self):
        """Returns a copy of this Face"""
        new_face = copy.copy(self)
        new_face.vertices = [Vertex(v.pos, duplicate=True) for v in self.vertices]
        new_face.edges = new_face.create_edges(duplicate=True)

        return new_face

    @property
    def center(self) -> PointType:
        """Center point of this face"""
        return np.average(self.points, axis=0)

    @property
    def normal(self) -> VectorType:
        """Returns a vector normal to this face.
        For non-planar faces the same rule as in OpenFOAM is followed:
        divide a quadrangle into 4 triangles, each joining at face center;
        a normal is the average of normals of those triangles."""
        # TODO: TEST
        # TODO: cache?
        points = self.points
        center = self.center

        side_1 = points - center
        side_2 = np.roll(points, -1, axis=0) - center
        normals = np.cross(side_1, side_2)

        return np.average(normals, axis=0)
