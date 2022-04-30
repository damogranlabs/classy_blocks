import numpy as np

from ..flat.face import Face
from ...util import functions as f

class Annulus:
    # In real-life, Annulus and Ring are the same 2D objects.
    # Here, however, Annulus is a 2D collection of faces whereas
    # Ring is an annulus that has been extruded to 3D.
    def __init__(self, center_point, outer_radius_point, normal, inner_radius, n_segments=None):
        self.center_point = np.asarray(center_point)
        self.normal = f.unit_vector(np.asarray(normal))
        self.outer_radius_point = np.asarray(outer_radius_point)
        self.outer_radius = f.norm(self.outer_radius_point - self.center_point)

        # tp keep consistent with Circle
        self.radius_point = self.outer_radius_point
        self.radius = self.outer_radius

        self.inner_radius = inner_radius

        self.inner_radius_point = self.center_point + \
            f.unit_vector(self.outer_radius_point - self.center_point)*self.inner_radius
        
        # TODO: TEST
        if n_segments is None:
            # 4 for rings, 8 for expanding shapes made from circles
            n_segments = 4
        self.n_segments = n_segments

        rot = lambda p, a: f.arbitrary_rotation(p, self.normal, a, self.center_point)
        angle = 2*np.pi/n_segments

        face = Face(
            [ # points
                self.inner_radius_point, self.outer_radius_point,
                rot(self.outer_radius_point, angle), rot(self.inner_radius_point, angle)
            ],
            [ # edges
                None, rot(self.outer_radius_point, angle/2),
                None, rot(self.inner_radius_point, angle/2)
            ]
        )

        # unlike with Circle, there's no core
        self.core_faces = None

        self.shell_faces = [
            face.rotate(self.normal, i*angle, self.center_point)
            for i in range(self.n_segments)
        ]

        # required by Shapes
        # TODO: is it?
        # TEST: is it?
        self.faces = self.shell_faces

    def translate(self, vector, **kwargs):
        # TODO: TEST
        return self.__class__(
            self.center_point + vector,
            self.radius_point + vector,
            self.normal,
            self.inner_radius, self.n_segments)
        
    def rotate(self, axis, angle, origin, **kwargs):
        # TODO: TEST
        r = lambda p: f.arbitrary_rotation(p, axis, angle, origin)

        return self.__class__(
            r(self.center_point),
            r(self.radius_point),
            f.arbitrary_rotation(self.normal, axis, angle, [0, 0, 0]),
            self.inner_radius, self.n_segments)

    def scale(self, outer_radius, inner_radius, **kwargs):
        # TODO: TEST
        return self.__class__(
            self.center_point,
            self.center_point + f.unit_vector(self.radius_point - self.center_point)*outer_radius,
            self.normal,
            inner_radius, self.n_segments)
    