from typing import List

import numpy as np

from classy_blocks.types import PointType, VectorType, NPPointType, NPVectorType
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.flat.disk import Disk, QuarterDisk
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.construct.shapes.round import RoundShape

from classy_blocks.util import constants
from classy_blocks.util import functions as f


def eighth_sphere_lofts(
    center_point: NPPointType,
    radius_point: NPPointType,
    normal: NPVectorType,
    geometry_name: str,
    diagonal_angle: float = np.pi / 5,
):
    """A collection of 4 lofts for an eighth of a sphere;
    used to construct all other sphere pieces and derivatives"""
    # An 8th of a sphere has 3 equal flat sides and one round;
    # the 'bottom' is the one perpendicular to given normal
    bottom = QuarterDisk(center_point, radius_point, normal)

    # rotate a QuarterDisk twice around 'bottom' two edges to get them;
    axes = {
        # around 'O-P1', 'front':
        "front": bottom.points["P1"] - bottom.points["O"],
        # around 'O-P3', 'left':
        "left": bottom.points["P3"] - bottom.points["O"],
        # diagonal O-D is obtained by rotating around an at 45 degrees
        "diagonal": f.rotate(bottom.points["P3"], normal, np.pi / 4, center_point) - bottom.points["O"],
    }

    front = bottom.copy().rotate(np.pi / 2, axes["front"], center_point)
    left = bottom.copy().rotate(-np.pi / 2, axes["left"], center_point)

    point_DU = f.rotate(bottom.points["D"], axes["diagonal"], -diagonal_angle, center_point)
    point_P2U = f.rotate(bottom.points["P2"], axes["diagonal"], -diagonal_angle, center_point)

    # 4 lofts for an eighth sphere, 1 core and 3 shell
    lofts: List[Loft] = []

    # core
    core = Loft(bottom.core[0], Face([front.points["S2"], front.points["D"], point_DU, left.points["D"]]))
    lofts.append(core)

    # shell
    shell_1 = Loft(bottom.shell[0], Face([front.points["D"], front.points["P2"], point_P2U, point_DU]))
    shell_1.project_side("right", geometry_name, edges=True)

    shell_2 = Loft(bottom.faces[2], Face([point_DU, point_P2U, left.points["P2"], left.points["D"]]))
    shell_2.project_side("right", geometry_name, edges=True)

    shell_3 = Loft(shell_1.top_face, left.faces[1])
    shell_3.project_side("right", geometry_name, edges=True)
    lofts += [shell_1, shell_2, shell_3]

    return lofts


class EighthSphere(RoundShape):
    """One eighth of a sphere, the base shape for half and whole spheres"""

    n_cores: int = 1

    def transform_function(self, **kwargs):
        # Nothing to do here
        pass

    def __init__(
        self, center_point: PointType, radius_point: PointType, normal: VectorType, diagonal_angle: float = np.pi / 5
    ):
        # TODO: move these to properties
        # BUG: move these to properties
        # (will not change with transforms!)
        self.center_point = np.asarray(center_point)
        self.radius_point = np.asarray(radius_point)
        self.normal = f.unit_vector(np.asarray(normal))

        self.lofts = eighth_sphere_lofts(
            self.center_point, self.radius_point, self.normal, self.geometry_name, diagonal_angle
        )

    ### Chopping
    def chop_axial(self, **kwargs):
        """Chop along given normal"""
        self.shell[0].chop(2, **kwargs)

    def chop_radial(self, **kwargs):
        """Chop along radius vector"""
        self.shell[0].chop(0, **kwargs)

    def chop_tangential(self, **kwargs):
        """Chop circumferentially"""
        for i, operation in enumerate(self.shell):
            if (i + 1) % 3 == 0:
                continue

            operation.chop(1, **kwargs)

    ### Patches
    def set_start_patch(self, name: str) -> None:
        # chop everything except the top loft,
        # that's the 3rd operation in shell
        for operation in self.core:
            operation.set_patch("bottom", name)

        for i, operation in enumerate(self.shell):
            if (i + 1) % 3 == 0:
                continue

            operation.set_patch("bottom", name)

    @property
    def operations(self):
        return self.lofts

    @property
    def core(self):
        return self.lofts[: self.n_cores]

    @property
    def shell(self):
        return self.lofts[self.n_cores :]

    @property
    def radius(self) -> float:
        """Radius of this sphere"""
        return f.norm(self.radius_point - self.center_point)

    @property
    def geometry_name(self) -> str:
        """Name of a unique geometry this will project to"""
        return f"sphere_{str(id(self))}"

    @property
    def geometry(self):
        return {
            self.geometry_name: [
                "type searchableSphere",
                f"origin {constants.vector_format(self.center_point)}",
                f"centre {constants.vector_format(self.center_point)}",
                f"radius {self.radius}",
            ]
        }


class Hemisphere(EighthSphere):
    """A Quarter of a sphere, used as a base for Hemisphere"""

    # TODO: TEST
    n_cores: int = 4

    def __init__(
        self, center_point: PointType, radius_point: PointType, normal: VectorType, diagonal_angle: float = np.pi / 5
    ):
        super().__init__(center_point, radius_point, normal, diagonal_angle)

        rotated_core = []
        rotated_shell = []

        for i in range(1, self.n_cores + 1):
            rotated_radius_point = f.rotate(self.radius_point, self.normal, i * np.pi / 2, self.center_point)

            rotated_eighth = eighth_sphere_lofts(
                self.center_point, rotated_radius_point, self.normal, self.geometry_name
            )

            rotated_core.append(rotated_eighth[0])
            rotated_shell += rotated_eighth[1:]

        self.lofts = rotated_core + rotated_shell

    @property
    def core(self):
        return self.lofts[: self.n_cores]

    @property
    def shell(self):
        return self.lofts[self.n_cores :]

    @classmethod
    def chain(cls, source, start_face=False):
        assert source.sketch_class == Disk

        # TODO: TEST
        if start_face:
            center_point = source.sketch_1.center
            radius_point = source.sketch_1.radius_point
            normal = -source.sketch_1.normal
        else:
            center_point = source.sketch_2.center
            radius_point = source.sketch_2.radius_point
            normal = source.sketch_2.normal

        return cls(center_point, radius_point, normal)
