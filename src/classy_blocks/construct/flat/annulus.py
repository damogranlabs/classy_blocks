from typing import List

import numpy as np

from classy_blocks.types import PointType, VectorType
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

        self.center_point = np.asarray(center_point)
        self.normal = f.unit_vector(np.asarray(normal))

        outer_radius_point = np.asarray(outer_radius_point)
        inner_radius_point = self.center_point + \
            f.unit_vector(outer_radius_point - self.center_point) * inner_radius

        self.n_segments = n_segments

        segment_angle = 2 * np.pi / n_segments

        face = Face([ # points
                inner_radius_point,
                outer_radius_point,
                f.rotate(outer_radius_point, self.normal, segment_angle, self.center_point),
                f.rotate(inner_radius_point, self.normal, segment_angle, self.center_point)
            ],
            [ # edges
                None, Origin(self.center_point),
                None, Origin(self.center_point)
            ]
        )

        self.core = []
        self.shell = [
            face.copy().rotate(i*segment_angle, self.normal, self.center_point)
            for i in range(self.n_segments)
        ]

    @property
    def faces(self) -> List[Face]:
        return self.shell
