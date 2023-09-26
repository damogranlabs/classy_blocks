import collections
import copy
from typing import List, Optional, Union

import numpy as np

from classy_blocks.base.element import ElementBase
from classy_blocks.base.exceptions import FaceCreationError
from classy_blocks.construct.edges import EdgeData, Line, Project
from classy_blocks.construct.point import Point
from classy_blocks.types import NPPointListType, NPPointType, NPVectorType, PointListType, PointType, ProjectToType
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
        points_shape = np.shape(points)
        if points_shape != (4, 3):
            raise FaceCreationError(
                "Provide exactly 4 points in 3D space",
                f"Available {points_shape[0]} points, each with {points_shape[1]} coordinates",
            )

        self.points = [Point(p) for p in points]
        # Edges
        self.edges: List[EdgeData] = [Line(), Line(), Line(), Line()]
        if edges is not None:
            if len(edges) != 4:
                raise FaceCreationError(
                    "Provide exactly 4 edges; use None for straight lines", f"Number of edges: {len(edges)}"
                )

            for i, edge in enumerate(edges):
                self.add_edge(i, edge)

        if check_coplanar:
            pts = self.point_array
            diff = abs(np.dot((pts[1] - pts[0]), np.cross(pts[3] - pts[0], pts[2] - pts[0])))
            if diff > constants.TOL:
                raise FaceCreationError(
                    "FacePoints are not coplanar!", f"Difference: {diff}, tolerance: {constants.TOL}"
                )

        # name of geometry this face can be projected to
        self.projected_to: Optional[str] = None
        # patch name to which this face can belong
        self.patch_name: Optional[str] = None

    def update(self, points: PointListType) -> None:
        """Moves points from current position to given"""
        for i, point in enumerate(points):
            self.points[i].position = np.array(point, dtype=constants.DTYPE)

    def add_edge(self, corner: int, edge_data: Union[EdgeData, None]) -> None:
        """Replaces an existing edge between corner and (corner+1);
        use None to delete an edge (replace with a straight line)"""
        if corner > 3:
            raise FaceCreationError("Provide a corner index between 0 and 3", f"Given corner index: {corner}")

        if edge_data is None:
            self.edges[corner] = Line()
        else:
            self.edges[corner] = edge_data

    def project_edge(self, corner: int, label: ProjectToType) -> None:
        """Adds a Project edge or add the label to an existing one"""
        edge = self.edges[corner]

        if isinstance(edge, Project):
            edge.add_label(label)
            return

        self.add_edge(corner, Project(label))

    def invert(self) -> "Face":
        """Reverses the order of points in this face."""
        self.points.reverse()

        return self

    def copy(self) -> "Face":
        """Returns a copy of this Face"""
        return copy.deepcopy(self)

    @property
    def point_array(self) -> NPPointListType:
        """A numpy array of this face's points"""
        return np.array([p.position for p in self.points])

    @property
    def center(self) -> NPPointType:
        """Center point of this face"""
        return np.average(self.point_array, axis=0)

    @property
    def normal(self) -> NPVectorType:
        """Returns a vector normal to this face.
        For non-planar faces the same rule as in OpenFOAM is followed:
        divide a quadrangle into 4 triangles, each joining at face center;
        a normal is the average of normals of those triangles."""
        points = self.point_array
        center = self.center

        side_1 = points - center
        side_2 = np.roll(points, -1, axis=0) - center
        normals = np.cross(side_1, side_2)

        return f.unit_vector(np.average(normals, axis=0))

    @property
    def parts(self):
        return self.points + self.edges

    def project(self, label: str, edges: bool = False, points: bool = False) -> None:
        """Project this face to given geometry;

        faces can only be projected to a single
        surface, therefore provide a single string
        (contrary to Edge/Vertex where 2 or even 3
        surfaces can be intersected and projected to).

        Use edges=True and points=True as a shortcut to
        also project face's edges and points to the same
        geometry. If you want more control (like projecting
        an edge to an intersection of two surfaces), use
        face.edges[0] = edges.Project(['label1', 'label2']).

        Geometry with provided label must be defined separately
        in Mesh object."""
        self.projected_to = label

        # TODO: TEST
        if edges:
            for i in range(4):
                self.project_edge(i, label)

        if points:
            for i in range(4):
                self.points[i].project(label)

    def shift(self, count: int) -> "Face":
        """Shifts points of this face by 'count', changing its
        starting point"""
        indexes = collections.deque(range(4))
        indexes.rotate(count)

        self.points = [self.points[i] for i in indexes]
        self.edges = [self.edges[i] for i in indexes]

        return self

    def reorient(self, start_near: PointType) -> "Face":
        """Shifts points of this face in circle so that the starting point
        is closest to given position; the normal is not affected."""
        position = np.array(start_near, dtype=constants.DTYPE)
        indexes = list(range(4))
        indexes.sort(key=lambda i: f.norm(position - self.points[i].position))

        self.shift(indexes[0])

        return self
