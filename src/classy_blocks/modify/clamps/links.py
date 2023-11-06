import abc

import numpy as np

from classy_blocks.items.vertex import Vertex
from classy_blocks.types import NPPointType, PointType, VectorType
from classy_blocks.util import functions as f


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


class RotationLink(LinkBase):
    """A link that maintains the same angular displacement
    between parent clamp/vertex and the linked one,
    around a given axis.

    It will only work correctly when leader is rotated
    around given axis and origin."""

    def __init__(self, leader: Vertex, follower: Vertex, origin: PointType, axis: VectorType):
        super().__init__(leader, follower)

        self.origin = np.array(origin)
        self.axis = np.array(axis)

        self.orig_leader_pos = np.copy(leader.position)
        self.orig_follower_pos = np.copy(follower.position)

    def transform(self) -> NPPointType:
        transform_angle = f.angle_between(self.leader.position, self.orig_leader_pos)

        return f.rotate(self.orig_follower_pos, transform_angle, self.axis, self.origin)
