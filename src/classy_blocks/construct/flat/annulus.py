from typing import List

import numpy as np

from classy_blocks.types import PointType, VectorType, NPPointType, NPVectorType
from classy_blocks.construct.edges import Origin
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.util import functions as f


class Annulus(Sketch):
    """A base for ring-like shapes;
    In real-life, Annulus and Ring are the same 2D objects.
    Here, however, Annulus is a 2D collection of faces whereas
    Ring is an annulus that has been extruded to 3D."""
    def __init__(self,
                 center_point:PointType,
                 outer_radius_point:PointType,
                 normal:VectorType,
                 inner_radius:float, n_segments:int=8):

        center_point = np.asarray(center_point)
        normal = f.unit_vector(np.asarray(normal))

        outer_radius_point = np.asarray(outer_radius_point)

        inner_radius_point = center_point + \
            f.unit_vector(outer_radius_point - center_point) * inner_radius

        segment_angle = 2 * np.pi / n_segments

        face = Face([ # points
                inner_radius_point,
                outer_radius_point,
                f.rotate(outer_radius_point, normal, segment_angle, center_point),
                f.rotate(inner_radius_point, normal, segment_angle, center_point)
            ],
            [ # edges
                None, Origin(center_point),
                None, Origin(center_point)
            ]
        )

        self.core = []
        self.shell = [
            face.copy().rotate(i*segment_angle, normal, center_point)
            for i in range(n_segments)
        ]

        assert self.inner_radius < self.outer_radius, "Outer ring radius must be larger than inner!"

    @property
    def faces(self) -> List[Face]:
        return self.shell

    # FIXME: do something with inner_/outer/_point/_vector confusion
    @property
    def inner_radius_point(self) -> NPPointType:
        """"""
        return self.faces[0].points[0]

    @property
    def outer_radius_point(self) -> NPPointType:
        """"""
        return self.faces[0].points[1]

    @property
    def radius_point(self) -> NPPointType:
        return self.outer_radius_point

    @property
    def radius(self) -> float:
        return f.norm(self.outer_radius_point - self.center)

    @property
    def center(self) -> NPPointType:
        """"""
        return self.faces[0].edges[1].origin

    @property
    def inner_radius(self) -> float:
        """Returns inner radius as length, that is, distance between
        center and inner radius point"""
        return f.norm(self.inner_radius_point - self.center)

    @property
    def outer_radius(self) -> float:
        """"""
        return f.norm(self.outer_radius_point - self.center)

    @property
    def normal(self) -> NPVectorType:
        return self.faces[0].normal

    @property
    def n_segments(self):
        return len(self.faces)
