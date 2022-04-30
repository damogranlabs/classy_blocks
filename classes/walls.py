from typing import List
import numpy as np

from ..classes.flat.annulus import Annulus
from ..classes.shapes import Elbow, Shape

from ..util import functions as f

class ElbowWall(Shape):
    inner_patch = 'left'

    sketch_class = Annulus

    def transform_function(self, **kwargs):
        return self.sketch_1 \
            .rotate(**kwargs) \
            .scale(**kwargs)

    def __init__(self,
        center_point_1:List, outer_radius_point_1:List, normal_1:List, thickness_1:float, 
        sweep_angle:float, arc_center:List, rotation_axis:List,
        outer_radius_2:float, thickness_2:float, n_segments:int=None):
        # TODO: TEST
        self.arguments = locals()

        outer_radius_1 = f.norm(np.asarray(outer_radius_point_1) - np.asarray(center_point_1))
        inner_radius_1 = outer_radius_1 - thickness_1
        inner_radius_2 = outer_radius_2 - thickness_2

        super().__init__(
            [center_point_1, outer_radius_point_1, normal_1, inner_radius_1, n_segments],
            {
                'axis': rotation_axis,
                'angle': sweep_angle,
                'origin': arc_center,
                'outer_radius': outer_radius_2,
                'inner_radius': inner_radius_2
            },
            {
                'axis': rotation_axis,
                'angle': sweep_angle/2,
                'origin': arc_center,
                'outer_radius': (outer_radius_1 + outer_radius_2)/2,
                'inner_radius': (inner_radius_1 + inner_radius_2)/2
            }
        )

        self.core = []
        self.shell = self.operations

    def set_inner_patch(self, patch_name):
        for s in self.shell:
            s.block.set_patch(self.inner_patch, patch_name)

    @classmethod
    def expand(cls, source:Elbow, thickness_1, thickness_2=None):
        # would work just fine as a method but is implemented as a class method
        # to be consistent with other expands and whatnot.
        # TODO: TEST
        if thickness_2 is None:
            thickness_2 = thickness_1

        s1 = source.sketch_1
        radius_point_1 = s1.center_point + \
            f.unit_vector(s1.radius_point - s1.center_point)*(s1.radius + thickness_1)

        return cls(
            source.sketch_1.center_point, radius_point_1, source.sketch_1.normal, thickness_1,
            source.arguments['sweep_angle'], source.arguments['arc_center'], source.arguments['rotation_axis'],
            source.sketch_2.radius + thickness_2, thickness_2, len(s1.shell_faces))

class FrustumWall(Shape):
    inner_patch = 'left'

    sketch_class = Annulus

    def transform_function(self, **kwargs):
        return self.sketch_1 \
            .translate(**kwargs) \
            .scale(**kwargs)

    def __init__(self, axis_point_1:List, axis_point_2:List,
        outer_radius_point_1:List, thickness_1:float,
        outer_radius_2:float, thickness_2:float,
        outer_radius_mid:float=None, n_segments=None):

        self.axis = np.asarray(axis_point_2) - np.asarray(axis_point_1)

        
        outer_radius_1 = f.norm(np.asarray(outer_radius_point_1) - np.asarray(axis_point_1))
        inner_radius_1 = outer_radius_1 - thickness_1
        inner_radius_2 = outer_radius_2 - thickness_2

        if outer_radius_mid is None:
            mid_params = None
        else:
            mid_params = {
                'vector': self.axis/2,
                'outer_radius': outer_radius_mid,
                'inner_radius': outer_radius_mid - (thickness_1 + thickness_2)/2
            }

        super().__init__(
            [axis_point_1, outer_radius_point_1, self.axis, inner_radius_1, n_segments],
            {
                'vector': self.axis,
                'inner_radius': inner_radius_2,
                'outer_radius': outer_radius_2
            },
            mid_params
        )

        self.shell = self.operations
        self.core = []

    @classmethod
    def expand(cls, thickness_1, thickness_2=None, thickness_mid=None):
        # TODO
        # TODO: TEST
        pass

class HemisphereWall(Shape):
    # TODO
    pass