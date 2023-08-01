from classy_blocks.items.vertex import Vertex
from classy_blocks.modify.clamps.clamp import ClampBase


class FreeClamp(ClampBase):
    def __init__(self, vertex: Vertex):
        self.vertex = vertex

    @property
    def params(self):
        return self.vertex.position

    @property
    def bounds(self):
        return [[-1, 1], [-1, 1], [-1, 1]]

    def update_params(self, params):
        self.vertex.move_to(params)

    @property
    def point(self):
        return self.vertex.position
