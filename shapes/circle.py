import numpy as np

from operations.base import Face

from util import geometry as g
from util import constants

class Circle:
    def __init__(self, center_point, radius_point, normal):
        """ creates 5 faces that serve as a base for other Shapes; """
        self.center_point = np.array(center_point)
        self.radius_point = np.array(radius_point)
        self.normal = np.array(normal)

        self.radius_vector = self.radius_point - self.center_point

        def rotate(p, angle):
            return g.arbitrary_rotation(p, self.normal, angle, self.center_point)

        # default settings
        self.vertex_ratio = constants.frustum_core_to_outer
        self.edge_ratio = constants.frustum_edge_to_outer

        # create faces
        # core
        vertex_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
        edge_angles = np.linspace(np.pi/4, 2*np.pi + np.pi/4, 4, endpoint=False)

        core_point = self.center_point + self.radius_vector*self.vertex_ratio
        core_points = [rotate(core_point, a) for a in vertex_angles]

        core_edge = self.center_point + self.radius_vector*self.edge_ratio
        core_edges = [rotate(core_edge, a) for a in edge_angles]

        self.core_face = Face(core_points, core_edges)

        # 4 shell faces around core
        shell_face_points = [core_points[0], self.radius_point, rotate(self.radius_point, np.pi/2), core_points[1]]
        shell_edge_points = [None, rotate(self.radius_point, np.pi/4), None, None]
        shell_face = Face(shell_face_points, shell_edge_points)

        self.shell_faces = [
            shell_face.rotate(self.normal, a, origin=self.center_point) for a in vertex_angles
        ]

    def translate(self, vector):
        vector = np.array(vector)
        
        center_point = self.center_point + vector
        radius_point = self.radius_point + vector
        
        # normal does not change during translation
        return Circle(center_point, radius_point, self.normal)

    def rotate(self, axis, angle, origin):
        # rotate center point and radius point around origin;
        # rotate normal around zero

        new_center_point = g.arbitrary_rotation(self.center_point, axis, angle, origin)
        new_radius_point = g.arbitrary_rotation(self.radius_point, axis, angle, origin)
        new_normal = g.arbitrary_rotation(self.normal, axis, angle, [0, 0, 0])

        return Circle(new_center_point, new_radius_point, new_normal)

    def scale(self, new_radius):
        radius_vector = self.radius_vector/g.norm(self.radius_vector) * new_radius
        new_radius_point = self.center_point + radius_vector

        # normal does not change during scaling
        return Circle(self.center_point, new_radius_point, self.normal)

