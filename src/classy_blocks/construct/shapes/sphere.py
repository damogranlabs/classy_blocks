from typing import List

import numpy as np

from classy_blocks.types import PointType, VectorType
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.flat.disk import QuarterDisk
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.construct.shapes.round import RoundShape

from classy_blocks.util import constants
from classy_blocks.util import functions as f

class EighthSphere(RoundShape):
    """One eighth of a sphere, the base shape for half and whole spheres"""
    def transform_function(self, **kwargs):
        # Nothing to do here
        pass

    def __init__(self,
                 center_point:PointType,
                 radius_point:PointType,
                 normal:VectorType,
                 diagonal_angle:float=np.pi/5):
        
        self.center_point = np.asarray(center_point)
        self.radius_point = np.asarray(radius_point)
        self.normal = f.unit_vector(np.asarray(normal))

        # An 8th of a sphere has 3 equal flat sides and one round;
        # the 'bottom' is the one perpendicular to given normal
        bottom = QuarterDisk(self.center_point, self.radius_point, self.normal)

        # rotate a QuarterDisk twice around 'bottom' two edges to get them;
        axes = {
            # around 'O-P1', 'front':
            'front': bottom.points['P1'] - bottom.points['O'],
            # around 'O-P3', 'left':    
            'left': bottom.points['P3'] - bottom.points['O'],
            # diagonal O-D is obtained by rotating around an at 45 degrees
            'diagonal': f.rotate(
                bottom.points['P3'], self.normal, np.pi/4, self.center_point) - \
                bottom.points['O']
        }

        front = bottom.copy().rotate(np.pi/2, axes['front'], self.center_point)
        left = bottom.copy().rotate(-np.pi/2, axes['left'], self.center_point)

        point_DU = f.rotate(bottom.points['D'], axes['diagonal'], -diagonal_angle, self.center_point)
        point_P2U = f.rotate(bottom.points['P2'], axes['diagonal'], -diagonal_angle, self.center_point)

        # 4 lofts for an eighth sphere, 1 core and 3 shell
        self.lofts:List[Loft] = []

        # core
        core = Loft(
            bottom.core[0], Face([
            front.points['S2'], front.points['D'], point_DU, left.points['D']])
        )
        self.lofts.append(core)

        # shell
        shell_1 = Loft(
            bottom.shell[0],
            Face([front.points['D'], front.points['P2'],  point_P2U, point_DU])
        )
        shell_1.project_side('right', self.geometry_name, edges=True)

        shell_2 = Loft(
            bottom.faces[2],
            Face([point_DU, point_P2U, left.points['P2'], left.points['D']
        ]))
        shell_2.project_side('right', self.geometry_name, edges=True)

        shell_3 = Loft(
            shell_1.top_face,
            left.faces[1]
        )
        shell_3.project_side('right', self.geometry_name, edges=True)
        self.lofts += [shell_1, shell_2, shell_3]

    ### Chopping
    def chop_axial(self, **kwargs):
        raise NotImplementedError("A sphere can only be chopped radially and tangentially")

    def chop_radial(self, **kwargs):
        self.shell[0].chop(0, **kwargs)

    def chop_tangential(self, **kwargs):
        for i, operation in enumerate(self.shell):
            if i+1 % 3 == 0:
                continue
            operation.chop(1, **kwargs)
            operation.chop(2, **kwargs)

    @property
    def operations(self):
        return self.lofts

    @property
    def core(self):
        return self.lofts[0]

    @property
    def shell(self):
        return self.lofts[1:]

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

# class QuarterSphere(EighthSphere):
#     """A Quarter of a sphere, used as a base for Hemisphere"""
#     def __init__(self,
#                  center_point: PointType,
#                  radius_point: PointType,
#                  normal: VectorType,
#                  diagonal_angle: float = np.pi / 5):
#         super().__init__(center_point, radius_point, normal, diagonal_angle)

#         copied_lofts = [
#             loft.copy().rotate(np.pi/2, self.normal, self.origin)
#             for loft in self.lofts
#         ]

#         # sort lofts so that the core is the first
#         self.lofts = [eighth_1.lofts[0], eighth_2.lofts[0]] + \
#             eighth_1.lofts[1:] + eighth_2.lofts[1:]

#     @property
#     def core(self):
#         return self.lofts[:2]

#     @property
#     def shell(self):
#         return self.lofts[2:]
