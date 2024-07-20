import numpy as np

from classy_blocks.construct.assemblies.assembly import Assembly
from classy_blocks.construct.shapes.cylinder import SemiCylinder, SlashedCylinder
from classy_blocks.types import PointType
from classy_blocks.util import functions as f


class TJoint(Assembly):
    def __init__(self, start_point: PointType, center_point: PointType, right_point: PointType, radius: float):
        start_point = np.array(start_point)
        center_point = np.array(center_point)
        right_point = np.array(right_point)

        start_vector = center_point - start_point
        right_vector = right_point - center_point

        left_point = center_point - right_vector

        radius_vector = radius * f.unit_vector(np.cross(right_vector, start_vector))

        # the right part
        start_right_sc = SlashedCylinder(start_point, center_point, start_point + radius_vector)
        right_sc = SlashedCylinder(right_point, center_point, right_point - radius_vector)

        # the left part
        start_left_sc = SlashedCylinder(start_point, center_point, start_point - radius_vector)
        left_sc = SlashedCylinder(left_point, center_point, left_point + radius_vector)

        # the top part
        top_right = SemiCylinder(center_point, right_point, center_point - radius_vector)
        top_left = SemiCylinder(center_point, left_point, center_point + radius_vector)

        shapes = [start_right_sc, right_sc, start_left_sc, left_sc, top_right, top_left]

        for shape in shapes:
            shape.remove_inner_edges(end=True)

        super().__init__(shapes)
