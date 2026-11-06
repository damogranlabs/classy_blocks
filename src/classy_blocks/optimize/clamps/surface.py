import numpy as np

from classy_blocks.cbtyping import NPPointType, PointType, VectorType
from classy_blocks.optimize.clamps.clamp import ClampBase
from classy_blocks.util import functions as f
from classy_blocks.util.constants import DTYPE


class PlaneClamp(ClampBase):
    """Clamp that restricts point movement
    during optimization to an infinite plane, defined by point and normal.

    Bounds are not supported."""

    def __init__(self, position: PointType, point: PointType, normal: VectorType):
        point = np.array(point, dtype=DTYPE)
        normal = f.unit_vector(normal)

        # build an in-plane basis from the coordinate axis least aligned with normal;
        # deterministic and always well-conditioned (angle to normal >= ~54.7 deg)
        ref_dir = np.eye(3, dtype=DTYPE)[np.argmin(np.abs(normal))]

        u_dir = f.unit_vector(np.cross(normal, ref_dir))
        v_dir = np.cross(normal, u_dir)  # already unit length and orthogonal

        self.normal = normal
        self.u_dir = u_dir
        self.v_dir = v_dir

        def position_function(params) -> NPPointType:
            return point + params[0] * u_dir + params[1] * v_dir

        super().__init__(position, position_function)

    @property
    def initial_guess(self):
        return [0, 0]


class ParametricSurfaceClamp(ClampBase):
    """Clamp that restricts point movement
    during optimization to a surface, defined by a function:

    p = f(u, v);

    Function f must take two parameters 'u' and 'v' and return a single point in 3D space."""

    @property
    def initial_guess(self):
        if self.initial_params is None:
            return [0, 0]

        return self.initial_params


# class InterpolatedSurfaceClamp(ClampBase):
#    TODO


# class TriangulatedSurfaceClamp(ClampBase):
#    TODO
