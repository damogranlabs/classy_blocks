from typing import Dict, List

import numpy as np

from classy_blocks.construct.edges import Origin
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.flat.sketches.sketch import Sketch
from classy_blocks.construct.point import Point
from classy_blocks.types import NPPointType, NPVectorType, PointType, VectorType
from classy_blocks.util import functions as f


class QuarterDisk(Sketch):
    """A base for shapes with quarter-circular
    cross-sections; a helper for creating SemiCircle and Circle;
    see description of Circle object for more details"""

    # ratios between core and outer points;
    # see docstring of Disk class
    diagonal_ratio = 0.7
    side_ratio = 0.62

    def __init__(self, center_point: PointType, radius_point: PointType, normal: VectorType):
        center_point = np.asarray(center_point)
        radius_point = np.asarray(radius_point)
        normal = f.unit_vector(np.asarray(normal))
        radius_vector = radius_point - center_point

        # calculate points needed to construct faces:
        # these points must not be remembered because they will not change
        # when transforming this sketch; they must be taken from the faces themselves
        points: Dict[str, NPPointType] = {
            "O": center_point,
            "S1": center_point + radius_vector * self.side_ratio,
            "P1": center_point + radius_vector,
            "D": center_point + radius_vector * self.diagonal_ratio,
        }
        points["D"] = f.rotate(points["D"], np.pi / 4, normal, center_point)
        points["P2"] = f.rotate(points["P1"], np.pi / 4, normal, center_point)
        points["S2"] = f.rotate(points["S1"], np.pi / 2, normal, center_point)
        points["P3"] = f.rotate(points["P1"], np.pi / 2, normal, center_point)

        def make_face(keys, edges):
            return Face([points[k] for k in keys], edges, check_coplanar=True)

        # core: 0-S-D-S
        self.core = [make_face(["O", "S1", "D", "S2"], None)]

        # shell 1: S1-P1-P2-D
        shell_face_1 = make_face(["S1", "P1", "P2", "D"], [None, Origin(center_point), None, None])

        # shell 2: D-P2-P3-S
        shell_face_2 = make_face(["D", "P2", "P3", "S2"], [None, Origin(center_point), None, None])

        self.shell = [shell_face_1, shell_face_2]

    @property
    def faces(self) -> List[Face]:
        return self.core + self.shell

    @property
    def radius_vector(self) -> NPVectorType:
        """Vector that points from center of this
        *Circle to its (first) radius point"""
        return self.radius_point - self.center

    @property
    def radius(self) -> float:
        """Radius of this *circle, length of self.radius_vector"""
        return float(f.norm(self.radius_vector))

    @property
    def radius_point(self) -> NPPointType:
        """Point at outer radius"""
        return self.points["P1"].position

    @property
    def center(self) -> NPPointType:
        """Center point of this sketch"""
        return self.points["O"].position

    @property
    def points(self) -> Dict[str, Point]:
        """Returns points as named during construction of a QuarterDisk"""
        # Refer to core and shell because SemiDisk and Disk will add new faces
        # to self.faces[]
        return {
            "O": self.core[0].points[0],
            "S1": self.core[0].points[1],
            "D": self.core[0].points[2],
            "S2": self.core[0].points[3],
            "P1": self.shell[0].points[1],
            "P2": self.shell[0].points[2],
            "P3": self.shell[1].points[2],
        }

    @property
    def n_segments(self):
        return len(self.shell)


class HalfDisk(QuarterDisk):
    """A base for shapes with semi-circular
    cross-sections; a helper for creating Circle;
    see description of Circle object for more details"""

    def __init__(self, center_point: PointType, radius_point: PointType, normal: VectorType):
        super().__init__(center_point, radius_point, normal)
        # rotate core and shell faces
        other_quarter = self.copy().rotate(np.pi / 2, self.normal, self.center)

        self.core = self.core + other_quarter.core
        self.shell = self.shell + other_quarter.shell


class Disk(HalfDisk):
    """A 2D sketch of an H-grid disk; to be used for
    all solid round shapes (cylinder, frustum, elbow, ...

    H-grid parameters:
    A quarter of a circle is created from 3 blocks;
    Central 'square' (0) and two curved 'rectangles' (1 and 2)

    P3
    |******* P2
    |  2    /**
    |      /    *
    S2----D      *
    |  0  |   1   *
    |_____S1______*
    O              P1

    Relative size of the inner square (O-D), diagonal_ratio:
    - too small will cause unnecessary high number of small cells in the square;
    - too large will prevent creating large numbers of boundary layers
    """

    def __init__(self, center_point: PointType, radius_point: PointType, normal: VectorType):
        super().__init__(center_point, radius_point, normal)
        # rotate core and shell faces
        other_half = self.copy().rotate(np.pi, self.normal, self.center)

        self.core = self.core + other_half.core
        self.shell = self.shell + other_half.shell
