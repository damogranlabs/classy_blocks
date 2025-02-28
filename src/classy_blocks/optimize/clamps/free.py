import numpy as np

from classy_blocks.cbtyping import PointType
from classy_blocks.optimize.clamps.clamp import ClampBase


class FreeClamp(ClampBase):
    def __init__(self, position: PointType):
        super().__init__(position, np.asarray)

    def get_params_from_vertex(self):
        """Returns parameters from initial vertex position"""
        # there's no need for all that math in super()
        return self.position

    @property
    def initial_guess(self):
        return self.position
