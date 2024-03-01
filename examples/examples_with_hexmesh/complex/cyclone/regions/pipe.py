from typing import List

import numpy as np
import parameters as params
from regions.region import Region

import classy_blocks as cb
from classy_blocks.types import NPPointType
from classy_blocks.util import functions as f


class Pipe(Region):
    def __init__(self, outer: Region):
        self.outer = outer

        faces: List[cb.Face] = []

        for operation in outer.elements:
            face = operation.get_closest_face(self.center)

            # filter faces that are from outer operations
            for point in face.point_array:
                if f.to_polar(point, axis="z")[0] > 1.01 * self.geo.r["pipe"]:
                    break
            else:
                faces.append(face)

        for face in faces:
            if np.dot(face.normal, face.center - self.center) > 0:
                face.invert()

        self.shell = cb.Shell(faces, self.geo.l["pipe"])

        # snap all points on top faces to outlet cylinder
        # and bottom faces to 'pipe' cylinder
        for operation in self.shell.operations:
            face = operation.top_face
            for point in face.points:
                polar = f.to_polar(point.position, axis="z")
                polar[0] = self.geo.r["outlet"]
                point.move_to(f.to_cartesian(polar, axis="z"))

            operation.project_side("bottom", "pipe", edges=True, points=True)

    @property
    def center(self) -> NPPointType:
        center_z = self.outer.elements[0].center[2]
        return f.vector(0, 0, center_z)

    @property
    def elements(self):
        return self.shell.operations

    def chop(self):
        self.shell.chop(length_ratio=0.5, start_size=params.BL_THICKNESS, c2c_expansion=params.C2C_EXPANSION)
        self.shell.chop(length_ratio=0.5, end_size=params.BL_THICKNESS, c2c_expansion=1 / params.C2C_EXPANSION)

    def project(self):
        for operation in self.elements:
            operation.project_side("top", "outlet", True, True)
