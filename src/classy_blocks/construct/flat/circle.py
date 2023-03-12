import copy
from typing import List, Dict

import numpy as np

from classy_blocks.types import PointType, VectorType, NPPointType, NPVectorType
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.construct.edges import Origin
from classy_blocks.util import functions as f

# ratios between core and outer points;
# see docstring of Circle class
CORE_DIAGONAL_RATIO = 0.7
CORE_SIDE_RATIO = 0.62

class QuarterCircle(Sketch):
    """A base for shapes with quarter-circular
    cross-sections; a helper for creating SemiCircle and Circle;
    see description of Circle object for more details"""
    # TODO: TEST
    def __init__(self,
                center_point:PointType,
                radius_point:PointType,
                normal:VectorType,
                diagonal_ratio:float=CORE_DIAGONAL_RATIO,
                side_ratio:float=CORE_SIDE_RATIO):
        self.center_point = np.asarray(center_point)
        self.radius_point = np.asarray(radius_point)
        self.normal = f.unit_vector(np.asarray(normal))

        self.diagonal_ratio = diagonal_ratio
        self.side_ratio = side_ratio

        # calculate points needed to construct faces:
        points:Dict[str, NPPointType] = {
            'O': self.center_point,
            'S1': self.center_point + self.radius_vector * self.side_ratio,
            'P1': self.center_point + self.radius_vector,
            'D': self.center_point + self.radius_vector * self.diagonal_ratio,
        }
        points['D'] = f.rotate(points['D'], self.normal, np.pi/4)
        points['P2'] = f.rotate(points['P1'], self.normal, np.pi/4)
        points['S2'] = f.rotate(points['S1'], self.normal, np.pi/2)
        points['P3'] = f.rotate(points['P1'], self.normal, np.pi/2)

        def make_face(keys, edges):
            return Face([points[k] for k in keys], edges)

        # core: 0-S-D-S
        self.core = [make_face(['O', 'S1', 'D', 'S2'], None)]

        # shell 1: S1-P1-P2-D
        shell_face_1 = make_face(
            ['S1', 'P1', 'P2', 'D'],
            [None, Origin(self.center_point), None, None]
        )

        # shell 2: D-P2-P3-S
        shell_face_2 = make_face(
            ['D', 'P2', 'P3', 'S2'],
            [None, Origin(self.center_point), None, None]
        )

        self.shell = [shell_face_1, shell_face_2]

    @property
    def faces(self) -> List[Face]:
        return self.core + self.shell

    @property
    def radius_vector(self) -> NPVectorType:
        """Vector that points from center of this
        *Circle to its (first) radius point"""
        return self.radius_point - self.center_point

    @property
    def radius(self) -> float:
        """Radius of this *circle, length of self.radius_vector"""
        return float(f.norm(self.radius_vector))

class SemiCircle(QuarterCircle):
    """A base for shapes with semi-circular
    cross-sections; a helper for creating Circle;
    see description of Circle object for more details"""
    def __init__(self,
                 center_point:PointType,
                 radius_point:PointType,
                 normal:VectorType,
                 diagonal_ratio:float=CORE_DIAGONAL_RATIO,
                 side_ratio: float=CORE_SIDE_RATIO):
        super().__init__(center_point, radius_point, normal, diagonal_ratio, side_ratio)
        # rotate core and shell faces
        other_quarter = self.copy().rotate(np.pi/2, self.normal, self.center_point)

        self.core = self.core + other_quarter.core
        self.shell = self.shell + other_quarter.shell

class Circle(SemiCircle):
    """A 2D sketch of an H-grid circle; to be used for
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
    Orthogonality of the inner square (O-S), side_ratio:
    """
    def __init__(self,
                 center_point:PointType,
                 radius_point:PointType,
                 normal:VectorType,
                 diagonal_ratio:float=CORE_DIAGONAL_RATIO,
                 side_ratio: float=CORE_SIDE_RATIO):
        super().__init__(center_point, radius_point, normal, diagonal_ratio, side_ratio)
        # rotate core and shell faces
        other_half = self.copy().rotate(np.pi/2, self.normal, self.center_point)

        self.core = self.core + other_half.core
        self.shell = self.shell + other_half.shell
