import dataclasses

from typing import ClassVar, Optional

import numpy as np

from classy_blocks.items.edge.base import Edge
from classy_blocks.types import PointType, VectorType
from classy_blocks.util import functions as f

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

    return f.arc_mid(axis, center, radius, edge_point_1, edge_point_2)

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