from typing import Callable, List, Optional

import numpy as np

from classy_blocks.items.vertex import Vertex
from classy_blocks.modify.clamps.clamp import ClampBase
from classy_blocks.types import NPPointType, PointType, VectorType
from classy_blocks.util import functions as f
from classy_blocks.util.constants import DTYPE


class PlaneClamp(ClampBase):
    """Clamp that restricts point movement
    during optimization to a plane, defined by point and normal."""

    def __init__(self, vertex: Vertex, point: PointType, normal: VectorType):
        point = np.array(point, dtype=DTYPE)
        normal = f.unit_vector(normal)

        # choose a vector that is not collinear with normal
        random_dir = f.unit_vector(normal + np.random.random(3))

        u_dir = f.unit_vector(np.cross(random_dir, normal))
        v_dir = f.unit_vector(np.cross(u_dir, normal))

        def position_function(params) -> NPPointType:
            return point + params[0] * u_dir + params[1] * v_dir

        super().__init__(vertex, position_function)

    @property
    def initial_params(self):
        return [0, 0]


class AnalyticSurfaceClamp(ClampBase):
    """Clamp that restricts point movement
    during optimization to a surface, defined by a function:

    p = f(u, v);

    Function f must take two parameters 'u' and 'v' and return a single point in 3D space."""

    def __init__(
        self, vertex: Vertex, function: Callable[[List[float]], NPPointType], initial: Optional[List[float]] = None
    ):
        if initial is None:
            initial = [0, 0]

        self._custom_initial = initial
        super().__init__(vertex, function)

    @property
    def initial_params(self):
        return self._custom_initial


class InterpolatedSurfaceClamp(ClampBase):
    pass


class TriangulatedSurfaceClamp(ClampBase):
    pass
