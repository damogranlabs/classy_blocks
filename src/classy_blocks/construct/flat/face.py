from typing import List, Optional

import copy

import numpy as np

from classy_blocks.types import VectorType, PointType, NPPointType PointListType
from classy_blocks.base.transformable import TransformableBase
from classy_blocks.data.edges import EdgeData
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
    def __init__(self, points:PointListType, edges:Optional[List[Optional[EdgeData]]]=None, check_coplanar:bool=False):
        points = np.asarray(points, dtype=constants.DTYPE)
        if np.shape(points) != (4, 3):
            raise ValueError("Provide exactly 4 points in 3D space")

        if check_coplanar:
            if abs(
                np.dot(
                    points[1] - points[0],
                    np.cross(points[3] - points[0], points[2] - points[0])
                )) > constants.TOL:
                raise ValueError("Points are not coplanar!")

            # TODO: coplanar edges?

        self.points = points

        if edges is None:
            self.edges = [None]*4
        else:
            self.edges = edges

        assert len(self.edges) == 4, "Provide exactly 4 edges; use None for straight lines"

    def translate(self, displacement: VectorType) -> 'Face':
        # TODO: do something with repetition all over translate/rotate/scale
        displacement = np.asarray(displacement, dtype=constants.DTYPE)

        self.points = np.array([
            p + displacement for p in self.points
        ], dtype=constants.DTYPE)

        for edge in self.edges:
            if edge is not None:
                edge.translate(displacement)

        return self

    def rotate(self, angle: float, axis: VectorType, origin: Optional[PointType] = None) -> 'Face':
        self.points = np.array([
            f.rotate(p, axis, angle, origin) for p in self.points
        ], dtype=constants.DTYPE)

        for edge in self.edges:
            if edge is not None:
                edge.rotate(angle, axis, origin)

        return self
    
    def scale(self, ratio: float, origin: Optional[PointType] = None) -> 'Face':
        if origin is None:
            origin = self.center

        self.points = np.array([
            f.scale(p, ratio, origin) for p in self.points
        ], dtype=constants.DTYPE)
        
        for edge in self.edges:
            if edge is not None:
                edge.scale(ratio, origin)
        
        return self

    def invert(self) -> None:
        """Reverses the order of points in this face."""
        self.points = np.flip(self.points, axis=0)
    
    def copy(self) -> 'Face':
        """Returns a copy of this Face"""
        return copy.deepcopy(self)

    @property
    def center(self) -> NPPointType:
        """Center point of this face"""
        return np.average(self.points, axis=0)

    @property
    def normal(self) -> NPVectorType:
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
