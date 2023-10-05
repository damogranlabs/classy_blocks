import numpy as np

from classy_blocks.items.vertex import Vertex
from classy_blocks.modify.clamps.clamp import ClampBase


class FreeClamp(ClampBase):
    def __init__(self, vertex: Vertex):
        super().__init__(vertex, np.asarray)

    def get_params_from_vertex(self):
        """Returns parameters from initial vertex position"""
        # there's no need for all that math in super()
        return self.vertex.position

    @property
    def initial_guess(self):
        return self.vertex.position
