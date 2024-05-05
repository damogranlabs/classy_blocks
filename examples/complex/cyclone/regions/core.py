from typing import List, Sequence

import numpy as np
import parameters as params
from regions.region import Region

import classy_blocks as cb
from classy_blocks.base.transforms import Transformation, Translation
from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.construct.shapes.round import RoundSolidShape
from classy_blocks.types import NPPointListType, NPVectorType, PointListType, PointType
from classy_blocks.util import functions as f


class DozenBlockDisk(Sketch):
    """A disk that has 3x3 inside quads and 12 outer
    See sketch for explanations"""

    layer_1_ratio = 0.25
    layer_2_ratio = 0.55

    face_map = [  # for creating blocks from points-by-layer
        [[0, 0], [0, 1], [0, 2], [0, 3]],  # core: 0
        [[0, 0], [1, 11], [1, 0], [1, 1]],  # layer 1: 1
        [[0, 0], [1, 1], [1, 2], [0, 1]],  # 2
        [[0, 1], [1, 2], [1, 3], [1, 4]],  # 3
        [[0, 1], [1, 4], [1, 5], [0, 2]],  # 4
        [[0, 2], [1, 5], [1, 6], [1, 7]],  # 5
        [[0, 2], [1, 7], [1, 8], [0, 3]],  # 6
        [[0, 3], [1, 8], [1, 9], [1, 10]],  # 7
        [[0, 3], [1, 10], [1, 11], [0, 0]],  # 8
        [[1, 0], [2, 0], [2, 1], [1, 1]],  # layer 3: 9
        [[1, 1], [2, 1], [2, 2], [1, 2]],  # 10
        [[1, 2], [2, 2], [2, 3], [1, 3]],  # 11
        [[1, 3], [2, 3], [2, 4], [1, 4]],  # 12
        [[1, 4], [2, 4], [2, 5], [1, 5]],  # 13
        [[1, 5], [2, 5], [2, 6], [1, 6]],  # 14
        [[1, 6], [2, 6], [2, 7], [1, 7]],  # 15
        [[1, 7], [2, 7], [2, 8], [1, 8]],  # 16
        [[1, 8], [2, 8], [2, 9], [1, 9]],  # 17
        [[1, 9], [2, 9], [2, 10], [1, 10]],  # 18
        [[1, 10], [2, 10], [2, 11], [1, 11]],  # 19
        [[1, 11], [2, 11], [2, 0], [1, 0]],  #  20
    ]

    neighbours = [  # for laplacian smoothing of the inside
        [15, 5, 1, 3],  # 0
        [0, 6, 8, 2],  # 1
        [9, 1, 11, 3],  # 2
        [14, 0, 2, 12],  # 3
        [16, 5, 15],  # 4
        [4, 17, 6, 0],  # 5
        [18, 7, 1, 5],  # 6
        [6, 19, 8],  # 7
        [1, 7, 20, 9],  # 8
        [2, 8, 21, 10],  # 9
        [22, 11, 9],  # 10
        [23, 10, 2, 12],  # 11
        [11, 24, 13, 3],  # 12
        [25, 12, 14],  # 13
        [13, 3, 15, 26],  # 14
        [14, 27, 4, 0],  # 15
    ]

    def _smooth_points(self, points: NPPointListType):
        # A very rudimentary 2D laplacian smoothing;
        # to be replaced with automatic neighbour search,
        # removing the need for this 'neighbours' map
        for i, nei_indexes in enumerate(self.neighbours):
            nei_points = np.take(points, nei_indexes, axis=0)
            points[i] = np.average(nei_points, axis=0)

    def __init__(self, perimeter: PointListType, center_point: PointType):
        self.perimeter = np.array(perimeter)
        center_point = np.asarray(center_point)

        # calculate 3 layers of points;
        # 1st layer, square, 4 points
        # 2nd layer, 3x3 squares, 9 points
        # 3rd layer, 12 shell faces, 12 points
        layer_2 = np.array([center_point + self.layer_2_ratio * (p - center_point) for p in self.perimeter])
        layer_1 = np.array([center_point + self.layer_1_ratio * (layer_2[i] - center_point) for i in (0, 3, 6, 9)])

        # Assemble a full list for smoothing:
        points_by_index = np.concatenate((layer_1, layer_2, self.perimeter))

        for _ in range(5):
            self._smooth_points(points_by_index)

        # reconstruct the list back to layer_1, layer_2, layer_3 for face creation
        points_by_layer = [points_by_index[:4], points_by_index[4:16], points_by_index[16:]]

        # the first point is layer index, the second is the point within layer
        self._faces: List[cb.Face] = []

        for face_indexes in self.face_map:
            face = cb.Face([points_by_layer[i[0]][i[1]] for i in face_indexes])
            self._faces.append(face)

        self.core = self._faces[:10]
        self.shell = self._faces[10:]

        for face in self.shell:
            face.add_edge(1, cb.Origin(center_point))

    @property
    def faces(self):
        return self._faces

    @property
    def center(self):
        return self.faces[0].center

    @property
    def normal(self) -> NPVectorType:
        return f.unit_vector(np.cross(self.perimeter[0] - self.center, self.perimeter[1] - self.center))

    @property
    def n_segments(self):
        return 12


class DozenBlockCylinder(RoundSolidShape):
    sketch_class = DozenBlockDisk

    def __init__(self, perimeter: PointListType, center_point: PointType, length: float):
        sketch = DozenBlockDisk(perimeter, center_point)
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
