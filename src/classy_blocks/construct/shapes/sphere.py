import numpy as np

from classy_blocks.cbtyping import NPPointType, NPVectorType, PointType, VectorType
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.flat.sketches.disk import QuarterDisk
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.construct.shape import Shape
from classy_blocks.util import constants
from classy_blocks.util import functions as f


def get_named_points(qdisk: QuarterDisk) -> dict[str, NPPointType]:
    """Returns a dictionary of named points for easier construction of sphere;
    points refer to QuarterDisk:

    P3
    |******* P2
    |  2    /**
    |      /    *
    S2----D      *
    |  0  |   1   *
    |_____S1______*
    O              P1
    """
    points = [face.point_array for face in qdisk.faces]

    return {
        "O": points[0][0],
        "P1": points[1][1],
        "P2": points[1][2],
        "P3": points[2][2],
        "S1": points[0][1],
        "S2": points[0][3],
        "D": points[0][2],
    }


def eighth_sphere_lofts(
    center_point: NPPointType,
    radius_point: NPPointType,
    normal: NPVectorType,
    geometry_label: str,
    diagonal_angle: float = np.pi / 5,
) -> list[Loft]:
    """A collection of 4 lofts for an eighth of a sphere;
    used to construct all other sphere pieces and derivatives"""
    # An 8th of a sphere has 3 equal flat sides and one round;
    # the 'bottom' is the one perpendicular to given normal
    bottom = QuarterDisk(center_point, radius_point, normal)

    bpoints = get_named_points(bottom)

    # rotate a QuarterDisk twice around 'bottom' two edges to get them;
    axes = {
        # around 'O-P1', 'front':
        "front": bpoints["P1"] - bpoints["O"],
        # around 'O-P3', 'left':
        "left": bpoints["P3"] - bpoints["O"],
        # diagonal O-D is obtained by rotating around an at 45 degrees
        "diagonal": f.rotate(bpoints["P3"], np.pi / 4, normal, center_point) - bpoints["O"],
    }

    front = bottom.copy().rotate(np.pi / 2, axes["front"], center_point)
    fpoints = get_named_points(front)
    left = bottom.copy().rotate(-np.pi / 2, axes["left"], center_point)
    lpoints = get_named_points(left)

    point_du = f.rotate(bpoints["D"], -diagonal_angle, axes["diagonal"], center_point)
    point_p2u = f.rotate(bpoints["P2"], -diagonal_angle, axes["diagonal"], center_point)

    # 4 lofts for an eighth sphere, 1 core and 3 shell
    lofts: list[Loft] = []

    # core
    core = Loft(
        bottom.core[0],
        Face([fpoints["S2"], fpoints["D"], point_du, lpoints["D"]]),
    )
    lofts.append(core)

    # shell
    shell_1 = Loft(bottom.shell[0], Face([fpoints["D"], fpoints["P2"], point_p2u, point_du]))
    shell_1.project_side("right", geometry_label, edges=True)

    shell_2 = Loft(bottom.faces[2], Face([point_du, point_p2u, lpoints["P2"], lpoints["D"]]))
    shell_2.project_side("right", geometry_label, edges=True)

    shell_3 = Loft(shell_1.top_face, left.faces[1])
    shell_3.project_side("right", geometry_label, edges=True)
    lofts += [shell_1, shell_2, shell_3]

    return lofts


class EighthSphere(Shape):
    """One eighth of a sphere, the base shape everything sphere-related"""

    n_cores: int = 1

    def __init__(
        self, center_point: PointType, radius_point: PointType, normal: VectorType, diagonal_angle: float = np.pi / 5
    ):
        center_point = np.asarray(center_point)
        radius_point = np.asarray(radius_point)
        normal = f.unit_vector(np.asarray(normal))

        rotated_core = []
        rotated_shell = []

        for i in range(self.n_cores):
            rotated_radius_point = f.rotate(radius_point, i * np.pi / 2, normal, center_point)

            rotated_eighth = eighth_sphere_lofts(
                center_point, rotated_radius_point, normal, self.geometry_label, diagonal_angle
            )

            rotated_core.append(rotated_eighth[0])
            rotated_shell += rotated_eighth[1:]

        self.lofts = rotated_core + rotated_shell

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
    def grid(self):
        return [self.core, self.shell]

    @property
    def radius_point(self) -> NPPointType:
        return self.shell[0].bottom_face.points[1].position

    @property
    def center_point(self) -> NPPointType:
        return self.lofts[0].bottom_face.points[0].position

    @property
    def normal(self) -> NPVectorType:
        return self.lofts[0].bottom_face.normal

    @property
    def radius(self) -> float:
        """Radius of this sphere"""
        return f.norm(self.radius_point - self.center_point)

    @property
    def geometry_label(self) -> str:
        """Name of a unique geometry this will project to"""
        return f"sphere_{id(self)}"

    @property
    def center(self):
        return self.center_point

    @property
    def geometry(self):
        return {
            self.geometry_label: [
                "type searchableSphere",
                f"origin {constants.vector_format(self.center_point)}",
                f"centre {constants.vector_format(self.center_point)}",
                f"radius {self.radius}",
            ]
        }


class QuarterSphere(EighthSphere):
    n_cores = 2


class Hemisphere(EighthSphere):
    # TODO: TEST
    n_cores: int = 4

    @classmethod
    def chain(cls, source, start_face=False):
        """Chain this sphere to the end face of a round solid shape;
        use start_face=True to chain to te start face."""
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

    def set_outer_patch(self, name):
        for operation in self.shell:
            operation.set_patch("right", name)
