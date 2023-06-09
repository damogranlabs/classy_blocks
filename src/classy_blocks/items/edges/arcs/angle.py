import dataclasses

import numpy as np

from classy_blocks.construct import edges
from classy_blocks.construct.point import Point
from classy_blocks.items.edges.arcs.arc_base import ArcEdgeBase
from classy_blocks.types import PointType, VectorType
from classy_blocks.util import functions as f


def arc_from_theta(edge_point_1: PointType, edge_point_2: PointType, angle: float, axis: VectorType) -> PointType:
    """Calculates a point on the arc edge from given sector angle and an
    axis of the arc. An interface to the Foundation's
    arc <vertex-1> <vertex-2> <angle> (axis) alternative edge specification:
    https://github.com/OpenFOAM/OpenFOAM-dev/commit/73d253c34b3e184802efb316f996f244cc795ec6

    Note: Meticulously transcribed from
    https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/mesh/blockMesh/blockEdges/arcEdge/arcEdge.C
    """
    if not (0 < abs(angle) < np.pi * 2):
        raise ValueError(f"Angle should be between 0 and 2*pi, got {angle}")

    axis = np.asarray(axis)
    edge_point_1 = np.asarray(edge_point_1)
    edge_point_2 = np.asarray(edge_point_2)

    dp = edge_point_2 - edge_point_1

    pm = (edge_point_1 + edge_point_2) / 2
    rm = f.unit_vector(np.cross(dp, axis))

    length = np.dot(dp, axis)

    chord = dp - length * axis
    mag_chord = f.norm(chord)

    center = pm - length * axis / 2 - rm * mag_chord / 2 / np.tan(angle / 2)
    radius = f.norm(edge_point_1 - center)

    return f.arc_mid(axis, center, radius, edge_point_1, edge_point_2)


@dataclasses.dataclass
class AngleEdge(ArcEdgeBase):
    """Alternative arc edge specification: sector angle and axis"""

    data: edges.Angle

    @property
    def third_point(self):
        return Point(
            arc_from_theta(self.vertex_1.position, self.vertex_2.position, self.data.angle, self.data.axis.components)
        )

    @property
    def description(self):
        # produce two lines
        # one with this edge's specification
        # arc <vertex-1> <vertex-2> <angle> (axis) alternative edge specification:
        out = f"\t// arc {self.vertex_1.index} {self.vertex_2.index} "
        out += f"{self.data.angle} {self.data.axis.description}\n"

        # the other is a classic three-point arc definition
        return out + super().description
