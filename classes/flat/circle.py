import numpy as np

from ...util import functions as f
from ...util import constants as c

from .face import Face

class Circle:
    def __init__(self, center_point, radius_point, normal, diagonal_ratio=None, side_ratio=None):
        self.center_point = np.asarray(center_point)
        self.radius_point = np.asarray(radius_point)
        self.normal = f.unit_vector(np.asarray(normal))

        self.radius_vector = self.radius_point - center_point
        self.radius = f.norm(self.radius_vector)

        self.n_segments = 8 # for expanding cylinders and stuff
        
        rot = lambda p, angle: f.arbitrary_rotation(p, self.normal, angle, center_point)

        # create faces
        # core: 4 faces
        core_angles = np.linspace(0, 2*np.pi, num=4, endpoint=False)

        # see constants for explanation of these ratios
        if diagonal_ratio is None:
            diagonal_ratio = c.circle_core_diagonal
        self.diagonal_ratio = diagonal_ratio

        if side_ratio is None:
            side_ratio = c.circle_core_side
        self.side_ratio = side_ratio
        
        # core points, 'D' and 'S'
        core_diagonal_point = center_point + self.radius_vector*self.diagonal_ratio
        core_side_point = center_point + self.radius_vector*self.side_ratio

        core_face = Face([
            self.center_point,
            core_side_point,
            rot(core_diagonal_point, np.pi/4),
            rot(core_side_point, 2*np.pi/4)
        ])

        self.core_faces = [
            core_face.rotate(self.normal, a, self.center_point)
            for a in core_angles            
        ]

        # shell faces around core
        shell_angles = np.linspace(0, 2*np.pi, num=4, endpoint=False)

        shell_face_1 = Face(
            [ # points
                core_face.points[1], self.radius_point,
                rot(self.radius_point, np.pi/4), core_face.points[2]
            ],
            [None,rot(self.radius_point, np.pi/8),None,None]
        )
        shell_face_2 = Face(
            [
                core_face.points[2], rot(self.radius_point, np.pi/4),
                rot(self.radius_point, np.pi/2), core_face.points[3]
            ],
            [None, rot(self.radius_point, 3*np.pi/8), None, None]
        )
        
        shell_faces_1 = [shell_face_1.rotate(self.normal, a, self.center_point) for a in shell_angles]
        shell_faces_2 = [shell_face_2.rotate(self.normal, a, self.center_point) for a in shell_angles]

        # combine shell_face_1 and shell_faces_2 in an alternating fashion
        # so that when traversing that list, faces are encountered in an orderly manner
        self.shell_faces = [None]*8
        self.shell_faces[::2] = shell_faces_1
        self.shell_faces[1::2] = shell_faces_2

        # required by all Shapes
        self.faces = self.core_faces + self.shell_faces

    def translate(self, vector, **kwargs):
        # TODO: TEST
        return self.__class__(
            self.center_point + vector,
            self.radius_point + vector,
            self.normal,
            self.diagonal_ratio, self.side_ratio)
            
    def rotate(self, axis, angle, origin, **kwargs):
        # TODO: TEST
        r = lambda p: f.arbitrary_rotation(p, axis, angle, origin)

        return self.__class__(
            r(self.center_point),
            r(self.radius_point),
            f.arbitrary_rotation(self.normal, axis, angle, [0, 0, 0]),
            self.diagonal_ratio, self.side_ratio)

    def scale(self, radius, **kwargs):
        # TODO: TEST
        r = lambda p: self.center_point + f.unit_vector(p - self.center_point)*radius
        
        return self.__class__(
            self.center_point,
            r(self.radius_point),
            self.normal,
            self.diagonal_ratio, self.side_ratio)
