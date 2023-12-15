from typing import Dict, List

import numpy as np
from scipy.spatial import ConvexHull

from classy_blocks.construct.operations.operation import Operation
from classy_blocks.types import NPPointListType, NPPointType, NPVectorType, OrientType, PointType
from classy_blocks.util import constants
from classy_blocks.util import functions as f


class DegenerateGeometryError(Exception):
    """Raised when orienting by ObserverSorter failed because of invalid geometry"""


class Triangle:
    """A 'Simplex' in scipy terms, but in 3D this is just a triangle."""

    def __init__(self, points: List[NPPointType]):
        self.points = points

    @property
    def normal(self) -> NPVectorType:
        side_1 = self.points[1] - self.points[0]
        side_2 = self.points[2] - self.points[0]

        return f.unit_vector(np.cross(side_1, side_2))

    @property
    def center(self) -> NPPointType:
        return np.average(self.points, axis=0)

    def flip(self):
        """Flips the triangle so that its normal points the other way"""
        self.points = np.flip(self.points, axis=0)

    def orient(self, hull_center: NPPointType) -> None:
        """Flips the triangle around (if needed) so that
        normal always points away from the provided hull center"""
        if np.dot(self.center - hull_center, self.normal) < 0:
            self.flip()


class Quadrangle:
    """A block face."""

    def __init__(self, triangles: List[Triangle]):
        if len(triangles) > 2:
            raise DegenerateGeometryError("A Quadrangle can only be defined with two triangles!")

        common_points = self.get_common_points(triangles[0].points, triangles[1].points)
        if len(common_points) != 2:
            raise DegenerateGeometryError("Two triangles that form a face do not have 2 common points!")

        unique_points = self.get_unique_points(triangles[0].points, triangles[1].points)
        if len(unique_points) != 2:
            raise DegenerateGeometryError("Two triangles that form a face do not have 2 unique points!")

        self.points = [*unique_points, *common_points]  # will be sorted later

    @staticmethod
    def get_unique_points(list_1: List[NPPointType], list_2: List[NPPointType]) -> List[NPPointType]:
        """Returns points from list_1 that are not in list_2"""
        common_points = Quadrangle.get_common_points(list_1, list_2)
        unique_points: List[NPPointType] = []

        for point in [*list_1, *list_2]:
            unique = True

            for common_point in common_points:
                if f.norm(point - common_point) < constants.TOL:
                    unique = False
                    break

            if unique:
                unique_points.append(point)

        return unique_points

    @staticmethod
    def get_common_points(list_1: List[NPPointType], list_2: List[NPPointType]) -> List[NPPointType]:
        """Returns points on the same position in both lists"""
        common_points: List[NPPointType] = []

        for point_1 in list_1:
            for point_2 in list_2:
                if f.norm(point_1 - point_2) < constants.TOL:
                    common_points.append(point_1)

        return common_points

    def get_common_point(self, quad_1: "Quadrangle", quad_2: "Quadrangle") -> NPPointType:
        """Identifies common points between this and two other quads."""
        common_1 = self.get_common_points(self.points, quad_1.points)
        common_2 = self.get_common_points(common_1, quad_2.points)

        if len(common_2) > 1:
            raise DegenerateGeometryError("More than a single common point between 3 faces!")

        return common_2[0]


class ViewpointReorienter:
    """Reorient an Operation so that faces are aligned as viewed by
    observer from a specified viewpoint.
    Two points must be specified, one 'in front' of the block (preferrably far away)
    and other 'above' the block (can also be far away).

    Will fail with degenerate hexahedras (concavity, wedges, dubiously aligned faces, ...).

    Reorienting will be done in-place so that all other Operation attributes
    remain unchanged. Therefore it is recommended you do sorting BEFORE adding
    any edges, patches, and so on. In other case, behaviour is undetermined."""

    def __init__(self, observer: PointType, ceiling: PointType):
        self.observer = np.array(observer)
        self.ceiling = np.array(ceiling)

    def _make_triangles(self, points: NPPointListType) -> List[Triangle]:
        """Creates triangles from hull's simplices"""
        hull = ConvexHull(points)
        center = np.average(points, axis=0)

        if len(hull.simplices) != 12:
            raise DegenerateGeometryError("The operation is not convex!")

        point_list = [np.take(points, indexes, axis=0) for indexes in hull.simplices]
        triangles = [Triangle(points) for points in point_list]

        for triangle in triangles:
            triangle.orient(center)

        return triangles

    def _get_normals(self, center: NPPointType) -> Dict[OrientType, NPVectorType]:
        v_observer = f.unit_vector(np.array(self.observer) - center)
        v_ceiling = f.unit_vector(np.array(self.ceiling) - center)

        # correct ceiling so that it's always at right angle with observer
        correction = np.dot(v_ceiling, v_observer) * v_observer
        v_ceiling -= correction
        v_ceiling = f.unit_vector(v_ceiling)

        v_left = f.unit_vector(np.cross(v_observer, v_ceiling))

        return {
            "front": v_observer,
            "back": -v_observer,
            "top": v_ceiling,
            "bottom": -v_ceiling,
            "left": v_left,
            "right": -v_left,
        }

    def _get_aligned(self, triangles: List[Triangle], vector: NPVectorType) -> List[Triangle]:
        return sorted(triangles, key=lambda t: np.dot(t.normal, vector))[-2:]

    def reorient(self, operation: Operation):
        triangles = self._make_triangles(operation.point_array)
        normals = self._get_normals(operation.center)

        remaining_triangles = set(triangles)

        quads: Dict[OrientType, Quadrangle] = {}

        for key, normal in normals.items():
            # Take two most nicely aligned triangles
            aligned = self._get_aligned(list(remaining_triangles), normal)

            quads[key] = Quadrangle(aligned)

            remaining_triangles -= set(aligned)

        # find each point by intersecting specific quads
        sorted_points = [
            quads["bottom"].get_common_point(quads["front"], quads["left"]),
            quads["bottom"].get_common_point(quads["front"], quads["right"]),
            quads["bottom"].get_common_point(quads["back"], quads["right"]),
            quads["bottom"].get_common_point(quads["back"], quads["left"]),
            quads["top"].get_common_point(quads["front"], quads["left"]),
            quads["top"].get_common_point(quads["front"], quads["right"]),
            quads["top"].get_common_point(quads["back"], quads["right"]),
            quads["top"].get_common_point(quads["back"], quads["left"]),
        ]

        for i, point in enumerate(operation.bottom_face.points):
            point.position = sorted_points[i]

        for i, point in enumerate(operation.top_face.points):
            point.position = sorted_points[i + 4]
