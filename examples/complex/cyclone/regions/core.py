from typing import ClassVar, List, Sequence

import numpy as np
import parameters as params
from regions.region import Region

import classy_blocks as cb
from classy_blocks.base.transforms import Transformation, Translation
from classy_blocks.cbtyping import NPVectorType, PointListType, PointType
from classy_blocks.construct.point import Point
from classy_blocks.construct.shapes.round import RoundSolidShape


class NineCoreDisk(cb.MappedSketch):
    """A disk that has 3x3 inside quads and 12 outer;
    see docs/cylinder.svg for a sketch"""

    quads: ClassVar = [
        # core
        [0, 1, 2, 3],  # 0
        # layer 1
        [0, 15, 4, 5],  # 1
        [0, 5, 6, 1],  # 2
        [1, 6, 7, 8],  # 3
        [1, 8, 9, 2],  # 4
        [2, 9, 10, 11],  # 5
        [2, 11, 12, 3],  # 6
        [3, 12, 13, 14],  # 7
        [3, 14, 15, 0],  # 8
        # layer 2
        [4, 16, 17, 5],  # 9
        [5, 17, 18, 6],  # 10
        [6, 18, 19, 7],  # 11
        [7, 19, 20, 8],  # 12
        [8, 20, 21, 9],  # 13
        [9, 21, 22, 10],  # 14
        [10, 22, 23, 11],  # 15
        [11, 23, 24, 12],  # 16
        [12, 24, 25, 13],  # 17
        [13, 25, 26, 14],  # 18
        [14, 26, 27, 15],  # 19
        [15, 27, 16, 4],  # 20
    ]

    def __init__(self, perimeter: PointListType, center_point: PointType):
        center_point = np.asarray(center_point)

        # inner points will be determined with smoothing;
        # a good enough starting estimate is the center
        # (anything in the same plane as other points)
        outer_points = np.asarray(perimeter)
        inner_points = np.ones((16, 3)) * np.average(outer_points, axis=0)

        positions = np.concatenate((inner_points, outer_points))

        self.center_point = Point(center_point)

        super().__init__(positions, self.quads)

        # smooth the inner_points (which are all invalid) into position
        smoother = cb.SketchSmoother(self)
        smoother.smooth()

    @property
    def core(self) -> List[cb.Face]:
        return self.faces[:9]

    @property
    def shell(self) -> List[cb.Face]:
        return self.faces[9:]

    def add_edges(self):
        for face in self.shell:
            face.add_edge(1, cb.Origin(self.center_point.position))

    @property
    def parts(self):
        return [*super().parts, self.center_point]

    @property
    def center(self):
        return self.center_point.position

    @property
    def grid(self):
        return [[self.faces[0]], self.faces[1:9], self.faces[9:]]

    @property
    def normal(self) -> NPVectorType:
        return self.faces[0].normal

    @property
    def n_segments(self):
        return 12


class DozenBlockCylinder(RoundSolidShape):
    sketch_class = NineCoreDisk

    def __init__(self, perimeter: PointListType, center_point: PointType, length: float):
        sketch = NineCoreDisk(perimeter, center_point)
        transforms: Sequence[Transformation] = [Translation(sketch.normal * length)]
        super().__init__(sketch, transforms)

    def chop_tangential(self, **kwargs):
        for operation in self.shell[:6]:
            operation.chop(1, **kwargs)

    def chop_radial(self, **kwargs):
        self.shell[0].chop(0, **kwargs)


class Core(Region):
    def __init__(self, points: PointListType):
        self.cylinder = DozenBlockCylinder(points, [0, 0, self.geo.z["upper"]], self.geo.l["lower"])

    def chop(self):
        self.cylinder.chop_radial(end_size=params.BL_THICKNESS, c2c_expansion=1 / params.C2C_EXPANSION)
        self.cylinder.chop_tangential(start_size=params.BULK_SIZE * self.geo.r["outlet"] / self.geo.r["body"])

    @property
    def elements(self):
        return self.cylinder.operations


class Outlet(Region):
    def __init__(self, core_cylinder: DozenBlockCylinder):
        self.operations = [
            cb.Extrude(op.bottom_face.copy().translate([0, 0, self.geo.l["outlet"]]), [0, 0, -self.geo.l["outlet"]])
            for op in core_cylinder.operations
        ]

    def chop(self):
        self.elements[0].chop(2, start_size=params.BULK_SIZE)

    @property
    def elements(self):
        return self.operations

    def set_patches(self):
        for operation in self.elements:
            operation.set_patch("bottom", "outlet")
