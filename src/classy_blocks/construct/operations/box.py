import numpy as np

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.types import PointType


class Box(Loft):
    """A Rudimentary Box with edges aligned to
    cartesian coordinates x-y-z. Refer to sketch
    in blockMesh documentation for explanation of args below:
    https://doc.cfd.direct/openfoam/user-guide-v6/blockmesh
    https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.3-mesh-generation-with-the-blockmesh-utility

    Args:
    - start_point: one corner of the box
    - diagonal_point: corner at the other end of volumetric diagonal to start_point;

    Box() will always sort input data so that it becomes aligned with
    cartesian coordinate system. Therefore edge 0-1 will correspond to x-axis,
    1-2 to y- and 0-4 to z-axis."""

    def __init__(self, start_point: PointType, diagonal_point: PointType):
        start_point = np.asarray(start_point)
        diagonal_point = np.asarray(diagonal_point)
        parr = np.vstack((start_point, diagonal_point)).T

        point_0 = np.asarray([min(parr[0]), min(parr[1]), min(parr[2])])
        point_6 = np.asarray([max(parr[0]), max(parr[1]), max(parr[2])])

        delta_x = [point_6[0] - point_0[0], 0, 0]
        delta_y = [0, point_6[1] - point_0[1], 0]
        delta_z = [0, 0, point_6[2] - point_0[2]]

        # this is a workaround to make linter happy (doesn't recognize numpy types properly?
        np_points = np.array([point_0, point_0 + delta_x, point_0 + delta_x + delta_y, point_0 + delta_y])

        bottom_face = Face(np_points)
        top_face = bottom_face.copy().translate(delta_z)

        super().__init__(bottom_face, top_face)
