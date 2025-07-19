from classy_blocks.base.exceptions import SideCreationError
from classy_blocks.cbtyping import OrientType
from classy_blocks.items.vertex import Vertex
from classy_blocks.util import constants


class Side:
    """A Block has 6 'sides', defined by vertices but addressed by OrientType;
    new faces can be created from those and patches are assigned to this"""

    def __init__(self, orient: OrientType, vertices: list[Vertex]):
        if len(vertices) != 8:
            raise SideCreationError("Pass exactly 8 of block vertices", f"Given {len(vertices)} vertice(s)")

        corners = constants.FACE_MAP[orient]
        self.vertices = [vertices[i] for i in corners]

        self._hash = hash(tuple(sorted([v.index for v in self.vertices])))

    def __eq__(self, other):
        return {v.index for v in self.vertices} == {v.index for v in other.vertices}

    def __hash__(self):
        return self._hash
