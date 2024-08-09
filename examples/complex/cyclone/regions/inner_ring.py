from typing import List

import numpy as np
import parameters as params
from regions.region import Region

import classy_blocks as cb
from classy_blocks.util import functions as f


class InnerRing(Region):
    """A ring, created by extruding innermost faces of given shapes"""

    radial_clamps = {60, 61, 62, 63, 64, 65}

    def _move_to_radius(self, point):
        polar = f.to_polar(point, axis="z")
        polar[0] = self.geo.r["pipe"]

        return f.to_cartesian(polar, axis="z")

    def _move_to_angle(self, point, angle):
        polar = f.to_polar(point, axis="z")
        polar[1] = angle

        return f.to_cartesian(polar, axis="z")

    def _reorient_face(self, face: cb.Face) -> None:
        # find the point with lowest z and lowest angle and
        # reorient face so that it starts with it
        # ACHTUNG! Not the prettiest piece of code.
        # TODO: Prettify
        z_min = 1e12
        angle_min = 1e12
        i_point = 5  # a.k.a. invalid

        for i, point in enumerate(face.point_array):
            polar = f.to_polar(point, axis="z")
            polar[1] += 10

            if polar[2] < z_min and polar[1] < angle_min:
                i_point = i
                angle_min = polar[1]
                z_min = polar[2]

        face.reorient(face.point_array[i_point])

        location = face.center
        normal = face.normal

        if np.dot(location, normal) > 0:
            face.invert()

    def __init__(self, lofts: List[cb.Loft]):
        center_point = f.vector(0, 0, self.geo.z["skirt"])
        outer_faces = [loft.get_closest_face(center_point) for loft in lofts]

        for face in outer_faces:
            self._reorient_face(face)

        inner_faces = [face.copy() for face in outer_faces]

        for face in inner_faces:
            for point in face.points:
                point.move_to(self._move_to_radius(point.position))

        self.lofts = [cb.Loft(outer_faces[i], inner_faces[i]) for i in range(len(inner_faces))]

    @property
    def elements(self):
        return self.lofts

    def chop(self):
        self.elements[0].chop(2, end_size=params.BL_THICKNESS, c2c_expansion=1 / params.C2C_EXPANSION)

    def project(self):
        for element in self.elements:
            element.project_side("top", "pipe", True, True)
