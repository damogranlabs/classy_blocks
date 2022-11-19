from typing import List, Optional
from classy_blocks import types
from numpy.typing import ArrayLike

import numpy as np

from classy_blocks.define.vertex import Vertex
from classy_blocks.define.edge import Edge
from classy_blocks.util import constants
from classy_blocks.util import functions as f
from classy_blocks.util import constants

class Face:
    def __init__(self, points: types.PointListType, check_coplanar: bool = False):
        """A Face is a collection of 4 Vertices and optionally 4 Edges,
        creating an arbitrary quadrangle. A Block can be made from 2 faces,
        connected with straight lines or, again, (optionally) curved edges."""
        points = np.asarray(points)
        if np.shape(points) != (4, 3):
            # TODO: TEST
            raise Exception("Provide exactly 4 points in 3D space")
        
        if check_coplanar:
            if abs(
                np.dot(
                    points[1] - points[0],
                    np.cross(points[3] - points[0], points[2] - points[0])
                )) > constants.tol:
                raise Exception("Points are not coplanar!")

            # TODO: coplanar edges?

        self.vertices:List[Vertex] = [Vertex(p) for p in points]

        self.edges:List[Edge] = [] # add with self.add_edge()

    def add_edge(self, index_1:int, points:types.EdgePointsType, kind:Optional[types.EdgeKindType]=None):
        """Specifies a non-line edge between a vertex specified by index_1 and the next one"""
        assert index_1 in (0, 1, 2, 3), f"Cannot assign index {index_1}; Face vertices only have indexes 0...3"

        self.edges.append(Edge(index_1, (index_1 + 1)%4, points, kind))

    def get_edges(self, top_face: bool = False) -> List[Edge]:
        if not top_face:
            return self.edges

        # if these edges refer to top face, correct the indexes
        return [Edge(
            e.block_index_1 + 4,
            e.block_index_2 + 4,
            e.points, e.kind) for e in self.edges]

    def translate(self, displacement:List):
        """Move this face by displacement vector.
        Returns the same face to enable chaining of transformations."""
        new_points = [v.translate(displacement).point for v in self.vertices]
        new_edges = [e.translate(displacement) for e in self.edges]
        
        new_face = Face(new_points)
        new_face.edges = new_edges

        return new_face

    def rotate(self, axis:List, angle:float, origin:List=None):
        """Rotate this face 'angle' around 'axis' going through 'origin'."""
        axis = np.asarray(axis)

        if origin is None:
            origin = self.center

        origin = np.asarray(origin)

        new_face = self.__class__([v.rotate(angle, axis, origin).point for v in self.vertices])
        new_face.edges = [e.rotate(angle, axis, origin) for e in self.edges]
        
        return new_face

    def scale(self, ratio:float, origin:List=None):
        """Scale with respect to given origin."""
        if origin is None:
            origin = self.center
        origin = np.asarray(origin)

        new_face = self.__class__([v.scale(ratio, origin).point for v in self.vertices])
        new_face.edges = [e.scale(ratio, origin) for e in self.edges]

        return new_face

    def invert(self):
        """Reverses the order of points in this face."""
        points = np.flip(self.points, axis=0)
        for i in range(len(self.vertices)):
            self.vertices[i].point = points[i]

    @property
    def points(self) -> ArrayLike:
        """Returns a list of points, extracted from vertices"""
        # TODO: cache?
        return np.array([v.point for v in self.vertices])

    @property
    def center(self) -> ArrayLike:
        """Returns the center point of this face."""
        # TODO: cache?
        return np.average(self.points, axis=0)

    @property
    def normal(self) -> ArrayLike:
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