from typing import List, Optional

import copy

import numpy as np

from classy_blocks.types import VectorType, PointType, PointListType, NPPointType, NPVectorType, NPPointListType
from classy_blocks.base.element import ElementBase
from classy_blocks.construct.edges import EdgeData, Line
from classy_blocks.util import constants
from classy_blocks.util import functions as f


class Face(ElementBase):
    """A collection of 4 Vertices and optionally 4 Edges,
    creating an arbitrary quadrangle.

    Args:
    - points: a list or a numpy array of exactly 4 points in 3d space
    - edges: an optional list of data for edge creation;
        if provided, it must be have exactly 4 elements,
        each element a list of data for edge creation; the format is
        the same as passed to Block.add_edge(). Each element of the list
        represents an edge between its corner and the next, for instance:

        edges=[None, Arc([0.4, 1, 1]]), None, None] will create an arc edge between the 1st and the 2nd vertex
        edges=[Project(['terrain']*4) will project all 4 edges
        of this face: 0-1, 1-2, 2-3, 3-0."""

    def __init__(
        self, points: PointListType, edges: Optional[List[Optional[EdgeData]]] = None, check_coplanar: bool = False
    ):
        # Points
        points = np.asarray(points, dtype=constants.DTYPE)
        if np.shape(points) != (4, 3):
            raise ValueError("Provide exactly 4 points in 3D space")

        self.points: NPPointListType = points

        # Edges
        self.edges: List[EdgeData] = [Line(), Line(), Line(), Line()]
        if edges is not None:
            assert len(edges) == 4, "Provide exactly 4 edges; use None for straight lines"

            for i, edge in enumerate(edges):
                if edge is not None:
                    self.edges[i] = edge

        if check_coplanar:
            pts = self.points
            assert (
                abs(np.dot((pts[1] - pts[0]), np.cross(pts[3] - pts[0], pts[2] - pts[0]))) < constants.TOL
            ), "FacePoints are not coplanar!"

    def translate(self, displacement: VectorType) -> "Face":
        displacement = np.asarray(displacement, dtype=constants.DTYPE)

        self.points = np.array([p + displacement for p in self.points], dtype=constants.DTYPE)

        for edge in self.edges:
            edge.translate(displacement)

        return self

    def rotate(self, angle: float, axis: VectorType, origin: Optional[PointType] = None) -> "Face":
        if origin is None:
            origin = self.center

        self.points = np.array([f.rotate(p, angle, axis, origin) for p in self.points], dtype=constants.DTYPE)

        for edge in self.edges:
            edge.rotate(angle, axis, origin)

        return self

    def scale(self, ratio: float, origin: Optional[PointType] = None) -> "Face":
        if origin is None:
            origin = self.center

        self.points = np.array([f.scale(p, ratio, origin) for p in self.points], dtype=constants.DTYPE)

        for edge in self.edges:
            edge.scale(ratio, origin)

        return self

    def invert(self) -> None:
        """Reverses the order of points in this face."""
        self.points = np.flip(self.points, axis=0)

    def copy(self) -> "Face":
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
        points = self.points
        center = self.center

        side_1 = points - center
        side_2 = np.roll(points, -1, axis=0) - center
        normals = np.cross(side_1, side_2)

        return f.unit_vector(np.average(normals, axis=0))
