import numpy as np
from typing import List

from ..classes.shapes import Shape
from ..classes.flat.circle import Semicircle
from ..classes.shapes import Cylinder

from ..util import constants as c
from ..util import functions as f


class Semicylinder(Cylinder):
    sketch_class = Semicircle

class SharpSemicylinder(Semicylinder):
    def transform_function(self, **kwargs):
        sketch_2 = self.sketch_1.translate(**kwargs)

        # TODO: replace by Semicircle.<someoperation>()
        # TODO: TEST
        origin = sketch_2.center_point
        axis = f.unit_vector(sketch_2.center_point - self.sketch_1.center_point)
        radius = sketch_2.radius_point - sketch_2.center_point
        normal = f.unit_vector(np.cross(axis, radius))

        def distance(point):
            return np.dot(point - origin, normal)

        def move(point):
            return axis*distance(point)
        
        for i, face in enumerate(sketch_2.faces):
            for j, point in enumerate(face.points):
                sketch_2.faces[i].points[j] += move(point)
            
            if face.edges is None:
                continue

            for j, edge in enumerate(face.edges):
                if edge is None:
                    continue

                face.edges[j] += move(edge)

        return sketch_2

class Joint:
    """ Connects a list of Shapes:
     - [Elbow/Frustum/Cylinder] or
     - [ElbowWall/FrustumWall/CylinderWall]
    """
    class Component:
        def __init__(self, shape:Shape, center_point:List[float]):
            self.shape = shape
            self.center_point = center_point

            self.near_point = None
            self.far_point = None

            # TODO: test
            # see which sketch is closer to specified center and assign
            # near and far points
            if f.norm(self.shape.sketch_1.center_point - self.center_point) > \
                f.norm(self.shape.sketch_2.center_point - self.center_point):
                # sketch_2 is closer to center
                self.near_point = self.shape.sketch_2.center_point
                self.far_point = self.shape.sketch_1.center_point
            else:
                self.near_point = self.shape.sketch_1.center_point
                self.far_point = self.shape.sketch_2.center_point

        @property
        def direction(self):
            return self.center_point - self.near_point


    def __init__(self, shapes:List[Shape], center_point:List[float]=None):
        assert len(shapes) >= 3, "At least 3 Shapes are needed to form a Joint"

        self.center_point = np.asarray(center_point)
        self.components = [Joint.Component(s, self.center_point) for s in shapes]


        





