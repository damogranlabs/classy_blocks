import warnings
from typing import List

from classy_blocks.base.curve import Curve
from classy_blocks.base.element import ElementBase
from classy_blocks.base.exceptions import EdgeCreationError
from classy_blocks.construct.point import Point, Vector
from classy_blocks.types import EdgeKindType, PointListType, PointType, ProjectToType, VectorType
from classy_blocks.util import functions as f


class EdgeData(ElementBase):
    """Common operations on classes for edge creation"""

    # edge kind, interpreted by edge factory;
    # coincident in some cases with what goes into blockMeshDict
    kind: EdgeKindType

    @property
    def parts(self):
        return []

    @property
    def center(self):
        warnings.warn("Transforming edge with a default center (0 0 0)!", stacklevel=2)
        return f.vector(0, 0, 0)

    def representation(self) -> EdgeKindType:
        return self.kind


class Line(EdgeData):
    """A 'line' edge is created by default and needs no extra parameters"""

    kind = "line"


class Arc(EdgeData):
    """Parameters for an arc edge: classic OpenFOAM circular arc
    definition with a single point lying anywhere on the arc"""

    kind = "arc"

    def __init__(self, arc_point: PointType):
        self.point = Point(arc_point)

    @property
    def parts(self):
        return [self.point]


class Origin(EdgeData):
    """Parameters for an arc edge, alternative ESI-CFD version;
    defined with an origin point and optional flatness (default 1)

    https://www.openfoam.com/news/main-news/openfoam-v20-12/pre-processing#x3-22000
    https://develop.openfoam.com/Development/openfoam/-/blob/master/src/mesh/blockMesh/blockEdges/arcEdge/arcEdge.H

    All arc variants are supported by classy_blocks;
    however, only the first (classic) one will be written to blockMeshDict for compatibility.
    If an edge was specified by 'angle' or 'origin', the definition will be output as a comment
    next to that edge definition."""

    kind = "origin"

    def __init__(self, origin: PointType, flatness: float = 1):
        self.origin = Point(origin)
        self.flatness = flatness

    @property
    def parts(self):
        return [self.origin]


class Angle(EdgeData):
    """Parameters for an arc edge, alternative definition
    by Foundation (.org); defined with sector angle and axis

    https://github.com/OpenFOAM/OpenFOAM-10/commit/73d253c34b3e184802efb316f996f244cc795ec6

    All arc variants are supported by classy_blocks;
    however, only the first (classic) one will be written to blockMeshDict for compatibility.
    If an edge was specified by 'angle' or 'origin', the definition will be output as a comment
    next to that edge definition."""

    kind = "angle"

    def __init__(self, angle: float, axis: VectorType):
        self.angle = angle
        self.axis = Vector(f.unit_vector(axis))

    def translate(self, displacement):
        """Axis is not to be translated"""

    def scale(self, ratio, origin=None):
        """Axis is not to be scaled"""

    @property
    def parts(self):
        return [self.axis]


class Spline(EdgeData):
    """Parameters for a spline edge"""

    kind = "spline"

    def __init__(self, points: PointListType):
        self.points: List[Point] = [Point(tr) for tr in points]

    @property
    def parts(self):
        return self.points


class PolyLine(Spline):
    """Parameters for a polyLine edge"""

    # a bug? (https://github.com/python/mypy/issues/8796)
    kind = "polyLine"  # type: ignore


class Project(EdgeData):
    """Parameters for a 'project' edge"""

    kind = "project"

    def __init__(self, label: ProjectToType):
        self.label = self.convert_label(label)
        self.check_length()

    @staticmethod
    def convert_label(label: ProjectToType) -> List[str]:
        """Makes sure label is always a list of strings
        of length 1 or 2"""
        if isinstance(label, str):
            return [label]

        # sort to keep consistent for debugging and testing purposes
        return list(sorted(label))

    def check_length(self) -> None:
        """Raises an exception if there are too many surfaces to project to"""
        if not (0 < len(self.label) < 3):
            raise EdgeCreationError(f"Edges can only be projected to 1 or 2 surfaces: {self.label}")

    def add_label(self, label: ProjectToType) -> None:
        """Projects this edge to another surface"""
        new_labels = self.convert_label(label)

        for add_label in new_labels:
            if add_label not in self.label:
                self.label.append(add_label)

        self.label.sort()
        self.check_length()


class OnCurve(EdgeData):
    """An edge, snapped to the given parametric curve"""

    kind = "curve"

    def __init__(self, curve: Curve, n_points: int = 10, representation: EdgeKindType = "spline"):
        self.curve = curve
        self.n_points = n_points

        self.representation = representation

    @property
    def parts(self):
        warnings.warn(f"Not transforming {self.__class__.__name__} edge!", stacklevel=2)
        return []

    @property
    def center(self):
        warnings.warn(f"Not transforming {self.__class__.__name__} edge!", stacklevel=2)
        return f.vector(0, 0, 0)
