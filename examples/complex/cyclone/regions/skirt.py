from regions.region import Region

import classy_blocks as cb


class Skirt(Region):
    """A region that connects inlet pipe's top faces to a ring on the outside of cyclone"""

    radial_clamps = (28, 23, 24, 30, 83, 81, 72, 66, 68, 69, 70, 84, 82, 80, 78, 76, 74, 71, 67)
    plane_clamps = (22, 25, 27, 29, 31, 34, 59, 51, 55, 47, 43, 39, 33, 37, 41, 45, 49, 53, 57)

    def __init__(self, inlet_shell: list[cb.Loft]):
        self.inlet_shell = inlet_shell

        # create 4 lofts, starting from inlet_shell's end faces, to a
        # plane, normal to z-axis
        z_coord = -self.geo.r["inlet"] - self.geo.l["skirt"]

        top_faces = [loft.top_face for loft in self.inlet_shell]
        bottom_faces = [face.copy() for face in top_faces]

        for face in bottom_faces:
            for point in face.points:
                point.position[2] = z_coord

        self.lofts = [cb.Loft(top_faces[i], bottom_faces[i]) for i in range(4)]

    @property
    def elements(self):
        return self.lofts

    def project(self):
        for operation in self.elements:
            operation.project_side("right", "body", True, True)
