from typing import List, Set

from regions.region import Region

import classy_blocks as cb
from classy_blocks.optimize.clamps.clamp import ClampBase


class Skirt(Region):
    """A region that connects inlet pipe's top faces to a ring on the outside of cyclone"""

    radial_clamps = {28, 24, 23, 30}
    plane_clamps = {22, 25, 27, 29, 31}

    def __init__(self, inlet_shell: List[cb.Loft]):
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

    def chop(self):
        # self.lofts[0].chop(axis=2, count=15)
        pass

    @property
    def elements(self):
        return self.lofts

    def get_radial_clamps(self, mesh: cb.Mesh) -> Set[ClampBase]:
        # Vertex 30 will be adjusted manually because
        # the round edge of inlet pipe at that spot will not be taken into consideration
        # when optimizing.
        angle = (self.geo.r["inlet"] + self.geo.l["skirt"]) / self.geo.r["body"]
        mesh.vertices[30].rotate(-angle, [0, 0, 1], [0, 0, 0])
        return super().get_radial_clamps(mesh)

    def project(self):
        for operation in self.elements:
            operation.project_side("right", "body", True, True)
