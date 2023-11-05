import abc

from classy_blocks.items.vertex import Vertex
from classy_blocks.types import NPPointType


class LinkBase(abc.ABC):
    """When optimizing a single vertex position,
    other vertices can be linked to it so that they move
    together with optimized vertex."""

    def __init__(self, leader: Vertex, follower: Vertex):
        self.leader = leader
        self.follower = follower

    def update(self) -> None:
        new_position = self.transform()
        self.follower.move_to(new_position)

    @abc.abstractmethod
    def transform(self) -> NPPointType:
        """Determine the new vertex position
        according to the type of link"""


class TranslationLink(LinkBase):
    """A link that maintains the same translation vector
    between parent clamp/vertex and the linked one."""

    def __init__(self, leader: Vertex, follower: Vertex):
        super().__init__(leader, follower)
        self.vector = self.follower.position - self.leader.position

    def transform(self) -> NPPointType:
        return self.leader.position + self.vector


class RotationLink:
    pass
