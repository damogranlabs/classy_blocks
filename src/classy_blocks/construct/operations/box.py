from typing import List

import numpy as np

from classy_blocks.types import PointType
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.util import functions as f

class Box(Loft):
    """A Rudimentary Box with edges aligned to
    cartesian coordinates x-y-z. Refer to sketch
    in blockMesh documentation for explanation of args below:
    https://doc.cfd.direct/openfoam/user-guide-v6/blockmesh
    https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.3-mesh-generation-with-the-blockmesh-utility

    Args:
    - point_0: 'bottom-left' corner position
    - sizes: a list of 3 floats for box sizes in x1, x2, x3 directions."""
    # TODO: change to corner+diagonal and auto-calculate coordinates
    def __init__(self, point_0: PointType, sizes:List[float]):
        point_0 = np.asarray(point_0)
        
        self.point_0 = point_0
        self.sizes = sizes

        delta_x = f.vector(1, 0, 0)*sizes[0]
        delta_y = f.vector(0, 1, 0)*sizes[1]
        delta_z = f.vector(0, 0, 1)*sizes[2]

        bottom_face = Face([
            point_0,
            point_0 + delta_x,
            point_0 + delta_x + delta_y,
            point_0 + delta_y
        ])
        top_face = bottom_face.copy().translate(delta_z)

        super().__init__(bottom_face, top_face)
