import abc

import numpy as np

from classy_blocks.cbtyping import NPPointType, NPVectorType, PointType, VectorType
from classy_blocks.util import constants
from classy_blocks.util import functions as f


class LinkBase(abc.ABC):
    """When optimizing a single vertex position,
    other vertices can be linked to it so that they move
    together with optimized vertex."""

    def __init__(self, leader: PointType, follower: PointType):
        self.leader = np.array(leader)
        self.follower = np.array(follower)

    def update(self) -> None:
        new_position = self.transform()
        self.follower = new_position

    @abc.abstractmethod
    def transform(self) -> NPPointType:
        """Determine the new vertex position
        according to the type of link"""

    def __str__(self):
        return f"Link {self.leader} - {self.follower}"


class TranslationLink(LinkBase):
    """A link that maintains the same translation vector
    between parent clamp/vertex and the linked one."""

    def __init__(self, leader: PointType, follower: PointType):
        super().__init__(leader, follower)
        self.vector = self.follower - self.leader

    def transform(self) -> NPPointType:
        return self.leader + self.vector


class RotationLink(LinkBase):
    """A link that maintains the same angular displacement
    between parent clamp/vertex and the linked one,
    around a given axis.

    It will only work correctly when leader is rotated
    around given axis and origin."""

    def __init__(self, leader: PointType, follower: PointType, axis: VectorType, origin: PointType):
        super().__init__(leader, follower)

        self.origin = np.array(origin)
        self.axis = f.unit_vector(axis)

        self.orig_leader_radius = self._get_radius(self.leader)
        self.orig_follower_pos = np.copy(self.follower)

        if f.norm(self.orig_leader_radius) < constants.TOL:
            raise ValueError("Leader and rotation axis are coincident!")

    def transform(self) -> NPPointType:
        prev_radius = self.orig_leader_radius
        this_radius = self._get_radius(self.leader)

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


class SymmetryLink(LinkBase):
    """A link that mirrors follower over a given plane."""

    def __init__(self, leader: PointType, follower: PointType, normal: VectorType, origin: PointType):
        self.normal = np.array(normal)
        self.origin = np.array(origin)

        super().__init__(leader, follower)
        self.transform()

    def _get_follower(self) -> NPPointType:
        return f.mirror(self.leader, self.normal, self.origin)

    def transform(self) -> NPPointType:
        return self._get_follower()
