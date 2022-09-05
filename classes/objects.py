import numpy as np
from typing import List

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
