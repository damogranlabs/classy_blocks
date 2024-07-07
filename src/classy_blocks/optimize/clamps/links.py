import abc

import numpy as np

from classy_blocks.items.vertex import Vertex
from classy_blocks.types import NPPointType, NPVectorType, PointType, VectorType
from classy_blocks.util import constants
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

    def __init__(self, leader: Vertex, follower: Vertex, axis: VectorType, origin: PointType):
        super().__init__(leader, follower)

        self.origin = np.array(origin)
        self.axis = f.unit_vector(axis)

        self.orig_leader_radius = self._get_radius(leader.position)
        self.orig_follower_pos = np.copy(self.follower.position)

        if f.norm(self.orig_leader_radius) < constants.TOL:
            raise ValueError("Leader and rotation axis are coincident!")

    def transform(self) -> NPPointType:
        prev_radius = self.orig_leader_radius
        this_radius = self._get_radius(self.leader.position)

        angle = f.angle_between(prev_radius, this_radius)

        cross_rad = np.cross(prev_radius, this_radius)
        if np.dot(cross_rad, self.axis) < 0:
            angle = -angle

        return f.rotate(self.orig_follower_pos, angle, self.axis, self.origin)

    def _get_height(self, point: NPPointType) -> NPVectorType:
        """Returns projection of the point to the axis"""
        return np.dot(point - self.origin, self.axis) * self.axis

    def _get_radius(self, point: NPPointType) -> NPVectorType:
        """Returns projection of the point to plane, given by origin and axis"""
        return (point - self.origin) - self._get_height(point)
