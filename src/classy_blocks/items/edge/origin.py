import dataclasses
import warnings

from typing import Callable, ClassVar

import numpy as np

from classy_blocks.items.edge.base import Edge
from classy_blocks.types import PointType
from classy_blocks.util import constants
from classy_blocks.util import functions as f

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
    return f.arc_mid(axis, center, radius, edge_point_1, edge_point_2)

@dataclasses.dataclass
class OriginEdge(Edge):
    """Alternative arc edge specification: origin and radius multiplier"""

    kind: ClassVar[str] = "origin"

    origin: PointType
    flatness: float = 1.0

    def transform(self, function: Callable):
        self.origin = function(self.origin)

        return self