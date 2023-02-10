"""Nonlinear block/face edges."""
import dataclasses

import abc
import warnings

from typing import Optional, Type, ClassVar, Callable

from classy_blocks.types import PointType, PointListType, VectorType
from classy_blocks.util import constants

import numpy as np

from classy_blocks.items.vertex import Vertex

from classy_blocks.util import functions as f
from classy_blocks.util import constants


def arc_mid(
    axis: VectorType, center: PointType, radius: float, edge_point_1: PointType, edge_point_2: PointType
) -> PointType:
    """Returns the midpoint of the specified arc in 3D space"""
    # Kudos to this guy for his shrewd solution
    # https://math.stackexchange.com/questions/3717427
    axis = np.asarray(axis)
    edge_point_1 = np.asarray(edge_point_1)
    edge_point_2 = np.asarray(edge_point_2)

    sec = edge_point_2 - edge_point_1
    sec_ort = np.cross(sec, axis)

    return center + f.unit_vector(sec_ort) * radius


def arc_from_theta(edge_point_1: PointType, edge_point_2: PointType, angle: float, axis: VectorType) -> PointType:
    """Calculates a point on the arc edge from given sector angle and an
    axis of the arc. An interface to the Foundation's
    arc <vertex-1> <vertex-2> <angle> (axis) alternative edge specification:
    https://github.com/OpenFOAM/OpenFOAM-dev/commit/73d253c34b3e184802efb316f996f244cc795ec6"""
    # Meticulously transcribed from
    # https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/mesh/blockMesh/blockEdges/arcEdge/arcEdge.C

    assert 0 < angle < 360, f"Angle {angle} should be between 0 and 2*pi"

    axis = np.asarray(axis)
    edge_point_1 = np.asarray(edge_point_1)
    edge_point_2 = np.asarray(edge_point_2)

    dp = edge_point_2 - edge_point_1

    pM = (edge_point_1 + edge_point_2) / 2
    rM = f.unit_vector(np.cross(dp, axis))

    l = np.dot(dp, axis)

    chord = dp - l * axis
    magChord = f.norm(chord)

    center = pM - l * axis / 2 - rM * magChord / 2 / np.tan(angle / 2)
    radius = f.norm(edge_point_1 - center)

    return arc_mid(axis, center, radius, edge_point_1, edge_point_2)


def arc_from_origin(
    edge_point_1: PointType,
    edge_point_2: PointType,
    center: PointType,
    adjust_center: bool = True,
    r_multiplier: float = 1.0,
):
    """Calculates a point on the arc edge from given endpoints and arc origin.
    An interface to ESI-CFD's alternative arc edge specification:
    arc <vertex-1> <vertex-2> origin [multiplier] (<point>)
    https://develop.openfoam.com/Development/openfoam/-/blob/master/src/mesh/blockMesh/blockEdges/arcEdge/arcEdge.H
    https://www.openfoam.com/news/main-news/openfoam-v20-12/pre-processing#pre-processing-blockmesh"""
    # meticulously transcribed from
    # https://develop.openfoam.com/Development/openfoam/-/blob/master/src/mesh/blockMesh/blockEdges/arcEdge/arcEdge.C

    # Position vectors from centre
    p1 = edge_point_1
    p3 = edge_point_2

    r1 = p1 - center
    r3 = p3 - center

    mag1 = f.norm(r1)
    mag3 = f.norm(r3)

    chord = p3 - p1

    axis = np.cross(r1, r3)

    # The average radius
    radius = 0.5 * (mag1 + mag3)

    # The included angle (not needed)
    # angle = np.arccos(np.dot(r1, r3)/(mag1*mag3))

    needs_adjust = False

    if adjust_center:
        needs_adjust = abs(mag1 - mag3) > constants.tol

        if r_multiplier != 1:
            # The min radius is constrained by the chord,
            # otherwise bad things will happen.
            needs_adjust = True
            radius = radius * r_multiplier
            radius = max(radius, (1.001 * 0.5 * f.norm(chord)))

    if needs_adjust:
        # The centre is not equidistant to p1 and p3.
        # Use the chord and the arcAxis to determine the vector to
        # the midpoint of the chord and adjust the centre along this
        # line.
        new_center = (0.5 * (p3 + p1)) + (radius**2 - 0.25 * f.norm(chord) ** 2) ** 0.5 * f.unit_vector(
            np.cross(axis, chord)
        )  # mid-chord -> centre

        warnings.warn("Adjusting center of edge between" + f" {str(edge_point_1)} and {str(edge_point_2)}")

        return arc_from_origin(p1, p3, new_center, False)

    # done, return the calculated point
    return arc_mid(axis, center, radius, edge_point_1, edge_point_2)


@dataclasses.dataclass
class Edge(abc.ABC):
    """Common stuff for all edge objects"""

    vertex_1: Vertex
    vertex_2: Vertex

    kind: ClassVar[str] = ""

    def transform(self, function: Callable):
        """An arbitrary transform of this edge by a specified function"""

    def translate(self, displacement: VectorType):
        """Move all points in the edge (but not start and end)
        by a displacement vector."""
        displacement = np.asarray(displacement)

        return self.transform(lambda p: p + displacement)

    def rotate(self, angle: float, axis: VectorType, origin: Optional[PointType] = None):
        """Rotates all points in this edge (except start and end Vertex) around an
        arbitrary axis and origin (be careful with projected edges, geometry isn't rotated!)"""
        if origin is None:
            origin = [0, 0, 0]

        return self.transform(lambda p: f.arbitrary_rotation(p, axis, angle, origin))

    def scale(self, ratio: float, origin):
        """Scales the edge points around given origin"""
        return self.transform(lambda p: Vertex.scale_point(p, ratio, origin))

    @property
    def is_valid(self) -> bool:
        """Returns True if this edge is elligible to be put into blockMeshDict"""
        # wedge geometries produce coincident
        # edges and vertices; drop those
        if f.norm(self.vertex_1.pos - self.vertex_2.pos) < constants.tol:
            return False

        # only arc edges need additional checking (blow-up 1/0 protection)
        # assume others valid
        return True


@dataclasses.dataclass
class ArcEdge(Edge):
    """Arc edge: defined by a single point"""

    kind: ClassVar[str] = "arc"
    arc_point: PointType

    def transform(self, function: Callable):
        self.arc_point = function(self.arc_point)

        return self

    @property
    def is_valid(self):
        if super().is_valid:
            # TODO: TEST
            # if case vertex1, vertex2 and point in between
            # are collinear, blockMesh will find an arc with
            # infinite radius and crash.
            # so, check for collinearity; if the three points
            # are actually collinear, this edge is redundant and can be
            # silently dropped
            
            # cross-product of three collinear vertices must be zero
            arm_1 = self.vertex_1.pos - self.arc_point
            arm_2 = self.vertex_2.pos - self.arc_point

            return abs(f.norm(np.cross(arm_1, arm_2))) > constants.tol

        return False


@dataclasses.dataclass
class OriginEdge(Edge):
    """Alternative arc edge specification: origin and radius multiplier"""

    kind: ClassVar[str] = "origin"

    origin: PointType
    flatness: float = 1.0

    def transform(self, function: Callable):
        self.origin = function(self.origin)

        return self


@dataclasses.dataclass
class AngleEdge(Edge):
    """Alternative arc edge specification: sector angle and axis"""

    kind: ClassVar[str] = "angle"

    angle: float
    axis: VectorType

    def rotate(self, angle: float, axis: VectorType, origin: Optional[PointType] = None):
        # this edge definition only needs to transform when rotating,
        # and that only for the axis
        self.axis = f.arbitrary_rotation(self.axis, axis, angle, origin)

        return self


@dataclasses.dataclass
class SplineEdge(Edge):
    """Spline edge, defined by multiple points"""

    kind: ClassVar[str] = "spline"

    points: PointListType

    def transform(self, function: Callable):
        self.points = np.array([function(p) for p in self.points])

        return self


@dataclasses.dataclass
class PolyLineEdge(SplineEdge):
    """PolyLine variant of SplineEdge"""

    kind: ClassVar[str] = "polyLine"


@dataclasses.dataclass
class ProjectedEdge(Edge):
    """Edge, projected to a specified geometry"""

    kind: ClassVar[str] = "project"

    geometry: str


class EdgeFactory:
    """Kind of a pattern"""

    def __init__(self):
        self.kinds = {}

    def register_kind(self, creator: Type[Edge]) -> None:
        """Introduces a new edge kind to this factory"""
        self.kinds[creator.kind] = creator

    def create(self, *args):
        """Creates an EdgeOps of the desired kind and returns it"""
        # all definitions begin with
        # vertex_1: Vertex
        # vertex_2: Vertex
        args = list(args)
        kind = self.kinds[args.pop(2)]
        return kind(*args)


factory = EdgeFactory()
factory.register_kind(ArcEdge)
factory.register_kind(OriginEdge)
factory.register_kind(AngleEdge)
factory.register_kind(SplineEdge)
factory.register_kind(PolyLineEdge)
factory.register_kind(ProjectedEdge)
