import numpy as np
from abc import ABC
from typing import List

from ..classes.operations import Loft, Revolve, Extrude
from ..classes.flat.face import Face
from ..classes.flat.circle import Circle
from ..classes.flat.annulus import Annulus

from ..util import constants as c
from ..util import functions as f

class Box(Extrude):
    def __init__(self, point_min:List, point_max:List):
        """ A box, aligned with coordinate system """
        base = Face([
            [point_min[0], point_min[1], point_min[2]],
            [point_max[0], point_min[1], point_min[2]],
            [point_max[0], point_max[1], point_min[2]],
            [point_min[0], point_max[1], point_min[2]],
        ], check_coplanar=True)

        extrude_vector = [0, 0, point_max[2] - point_min[2]]

        super().__init__(base, extrude_vector)

class Shape(ABC):
    """ An object, lofted from 3 cross-sections (sketches),
    each transformed by functions, specified in inherited classes """
    axial_axis = 2
    radial_axis = 0
    tangential_axis = 1

    bottom_patch = 'bottom'
    top_patch = 'top'
    outer_patch = 'right'

    sketch_class = None # Circle/Annulus/whatever, defined in inherited classes

    def transform_function(self, **kwargs):
        # a function that transforms sketch_1 to sketch_2;
        # from faces of these two a Loft will be made.
        # to be implemented in inherited classes
        pass

    def __init__(self, args_1, transform_2_args, transform_mid_args=None):
        # start with sketch_1 and transform it
        # using self.transform_function(transform_2_args) to obtain sketch_2;
        # use self.transform_function(transform_mid_args) to obtain mid sketch
        # (only if applicable)
        self.sketch_1 = self.sketch_class(*args_1)
        self.sketch_2 = self.transform_function(**transform_2_args)

        # TODO: TEST
        if transform_mid_args is not None:
            self.sketch_mid = self.transform_function(**transform_mid_args)
            loft_edges = [face.points for face in self.sketch_mid.faces]
        else:
            self.sketch_mid = None
            loft_edges = [None]*len(self.sketch_1.faces)

        self.operations = [
            Loft(
                self.sketch_1.faces[i],
                self.sketch_2.faces[i],
                loft_edges[i]
            ) for i in range(len(self.sketch_1.faces))
        ]

    @property
    def blocks(self):
        return [o.block for o in self.operations]
    
    ### Patches
    def set_patch(self, side, patch_name):
        for b in self.blocks:
            b.set_patch(side, patch_name)

    def set_bottom_patch(self, patch_name):
        self.set_patch(self.bottom_patch, patch_name)

    def set_top_patch(self, patch_name):
        self.set_patch(self.top_patch, patch_name)
    
    def set_outer_patch(self, patch_name):
        for s in self.shell:
            s.block.set_patch(self.outer_patch, patch_name)

    ### Chopping
    def chop_axial(self, **kwargs):
        self.operations[0].chop(self.axial_axis, **kwargs)

    def chop_radial(self, **kwargs):
        # scale all radial sizes to this ratio or core cells will be
        # smaller than shell's
        c2s_ratio = max(c.circle_core_diagonal, c.circle_core_side)
        if 'start_size' in kwargs:
            kwargs['start_size'] *= c2s_ratio
        if 'end_size' in kwargs:
            kwargs['end_size'] *= c2s_ratio

        self.shell[0].chop(self.radial_axis, **kwargs)

    def chop_tangential(self, **kwargs):
        for s in self.shell:
            s.chop(self.tangential_axis, **kwargs)
    
    def set_cell_zone(self, cell_zone):
        for b in self.blocks:
            b.cell_zone = cell_zone

class Elbow(Shape):
    """ A curved round shape of varying cross-section """
    sketch_class = Circle

    def transform_function(self, **kwargs):
        return self.sketch_1 \
            .rotate(**kwargs) \
            .scale(**kwargs)

    def __init__(self, center_point_1:List, radius_point_1:List, normal_1:List,
        sweep_angle:float, arc_center:List, rotation_axis:List, radius_2:float):
        self.arguments = locals()

        radius_1 = f.norm(np.asarray(radius_point_1) - np.asarray(center_point_1))

        super().__init__(
            [center_point_1, radius_point_1, normal_1], # arguments for this cross-section
            { # transformation parameters for sketch_2
                # parameters for .rotate
                'axis': rotation_axis,
                'angle': sweep_angle,
                'origin': arc_center,
                # parameters for .scale
                'radius': radius_2
            },
            { # transform parameters for mid sketch
                'axis': rotation_axis,
                'angle': sweep_angle/2,
                'origin': arc_center,
                'radius': (radius_1 + radius_2)/2
            }
        )

        self.core = self.operations[:4]
        self.shell = self.operations[4:]

    @classmethod
    def chain(cls, source, sweep_angle:float, arc_center:List, rotation_axis:List, radius_2:float, start_face=False):
        # TODO: TEST
        """ Use another round Shape's end face as a starting point for this Elbow;
        Returns a new Elbow object. To start from the other side, use start_face = True """
        assert source.sketch_class == Circle

        # TODO: TEST
        if start_face:
            s = source.sketch_1
        else:
            s = source.sketch_2
        
        return cls(
            s.center_point,
            s.radius_point,
            s.normal,
            sweep_angle,
            arc_center,
            rotation_axis,
            radius_2)


class Frustum(Shape):
    sketch_class = Circle

    def transform_function(self, **kwargs):
        return self.sketch_1 \
            .translate(**kwargs) \
            .scale(**kwargs)

    def __init__(self, axis_point_1:List, axis_point_2:List, radius_point_1:List, radius_2:float, radius_mid=None):
        """ Creates a cone frustum (truncated cylinder) with axis between points
        'axis_point_1' and 'axis_point_2'.
        'radius_point_1' defines starting point for blocks and radius_2 defines end radius
        (NOT A POINT! 'radius_point_2' is calculated from the other 3 points so that
        all four lie on the same plane).

        Sides are straight unless radius_mid is given; in that case a profiled body
        of revolution is created. """
        self.arguments = locals()
        self.axis = np.asarray(axis_point_2) - np.asarray(axis_point_1)

        if radius_mid is None:
            mid_params = None
        else:
            mid_params = {
                'vector': self.axis/2,
                'radius': radius_mid
            }

        super().__init__(
            [axis_point_1, radius_point_1, self.axis],
            {
                'vector': self.axis,
                'radius': radius_2
            },
            mid_params
        )

        self.core = self.operations[:4]
        self.shell = self.operations[4:]
    
    @classmethod
    def chain(cls, source, length, radius_2, radius_mid=None):
        """ Chain this Frustum to an existing Shape;
        Use length < 0 to begin on start face and go 'backwards' """
        # TODO: TEST
        assert source.sketch_class == Circle

        if length < 0:
            sketch = source.sketch_1
        else:
            sketch = source.sketch_2

        axis_point_2 = sketch.center_point + f.unit_vector(sketch.normal)*length
        
        return cls(sketch.center_point, axis_point_2, sketch.radius_point, radius_2, radius_mid)

class Cylinder(Frustum):
    def __init__(self, axis_point_1:List, axis_point_2:List, radius_point_1:List):
        self.arguments = locals()
        """ a Frustum with constant radius """
        radius_1 = f.norm(np.array(radius_point_1) - np.array(axis_point_1))

        super().__init__(axis_point_1, axis_point_2, radius_point_1, radius_1)

    @classmethod
    def chain(cls, source, length):
        """ Creates a new Cylinder on start or end face of a round Shape (Elbow, Frustum, Cylinder);
        Use length > 0 to extrude 'forward' from source's end face;
        Use length < 0 to extrude 'backward' from source' start face """
        # TODO: TEST
        if length > 0:
            sketch = source.sketch_2
        else:
            sketch = source.sketch_1

        axis_point_1 = sketch.center_point
        radius_point_1 = sketch.radius_point
        normal = sketch.normal

        axis_point_2 = axis_point_1 + f.unit_vector(normal)*length

        return cls(axis_point_1, axis_point_2, radius_point_1)

class RevolvedRing(Shape):
    """ A ring specified by its cross-section; can be of arbitrary shape.
    Points must be specified in the following order:
            p3---___
           /        ---p2
          /              \\
         p0---------------p1
           
    0---- -- ----- -- ----- -- ----- -- --->> axis

    In this case, chop_*() will work as intended, otherwise
    the axes will be swapped or blocks will be inverted.

    Because of RevolvedRing's arbitrary shape, there is no
    'start' or 'end' sketch and .expand()/.contract() methods
    are not available. """
    axial_axis = 0
    radial_axis = 1
    tangential_axis = 2

    bottom_patch = 'left'
    top_patch = 'right'
    inner_patch = 'front'
    outer_patch = 'back'

    def __init__(self, axis_point_1, axis_point_2, points, edges=[None]*4, n_segments=4):
        self.arguments = locals()

        assert len(points) == 4
        assert len(edges) == 4

        self.axis_point_1 = np.asarray(axis_point_1)
        self.axis_point_2 = np.asarray(axis_point_2) 
        self.axis = self.axis_point_2 - self.axis_point_1
        self.center_point = self.axis_point_1

        self.n_segments = n_segments

        self.points = points
        self.edges = edges

        face = Face(self.points, self.edges)
        angle = 2*np.pi/self.n_segments

        self.operations = [
            Revolve(
                face.rotate(self.axis, angle*i, self.center_point),
                angle, self.axis, self.axis_point_1
            ) for i in range(self.n_segments)
        ]

        self.core = []
        self.shell = self.operations

    def set_inner_patch(self, patch_name):
        for s in self.shell:
            s.block.set_patch(self.inner_patch, patch_name)

class ExtrudedRing(Shape):
    sketch_class = Annulus

    inner_patch = 'left'

    def transform_function(self, **kwargs):
        return self.sketch_1 \
            .translate(**kwargs)
    
    """ A revolved ring but specified like a Cylinder """
    def __init__(self, axis_point_1:List, axis_point_2:List,
        outer_radius_point_1:List, inner_radius:float, n_segments=8):
        self.arguments = locals()
        self.axis = np.asarray(axis_point_2) - np.asarray(axis_point_1)

        super().__init__(
            [axis_point_1, outer_radius_point_1, self.axis, inner_radius, n_segments],
            {
                'vector': self.axis
            },
            None
        )

        self.core = []
        self.shell = self.operations
    
    @classmethod
    def chain(cls, source, length):
        """ Creates a new cylinder on start or end face of a round source Shape;
        Use length > 0 to go 'forward' from source's end face or
        length < 0 to go 'backward' from source's start face """
        # TODO: TEST
        if length > 0:
            s = source.sketch_2
        else:
            s = source.sketch_1

        return cls(
            s.center_point,
            s.center_point + f.unit_vector(s.normal)*length,
            s.outer_radius_point, 
            s.inner_radius,
            n_segments=s.n_segments
        )

    @classmethod
    def expand(cls, source, thickness:float):
        """ Create a new concentric Ring with radius, enlarged by 'thickness';
        Can be used on Cylinder or ExtrudedRing """
        # TODO: TEST
        s1 = source.sketch_1
        s2 = source.sketch_2

        new_radius_point = s1.center_point + \
            f.unit_vector(s1.radius_point - s1.center_point)* \
            (s1.radius + thickness)

        return cls(
            s1.center_point,
            s2.center_point,
            new_radius_point,
            s1.radius,
            n_segments=s1.n_segments)
    
    @classmethod
    def contract(cls, source, inner_radius:float):
        # TODO: TEST
        assert source.__class__ == cls
        
        s1 = source.sketch_1
        s2 = source.sketch_2
        assert inner_radius < s1.inner_radius

        return cls(
            s1.center_point,
            s2.center_point,
            s1.inner_radius_point,
            inner_radius
        )

    def set_inner_patch(self, patch_name):
        for s in self.shell:
            s.block.set_patch(self.inner_patch, patch_name)

class Hemisphere(Shape):
    sketch_class = None # this is a special-ish case

    def __init__(self, center_point, radius_point, normal):
        self.arguments = locals()
        self.center_point = np.asarray(center_point)
        self.radius_point = np.asarray(radius_point)
        self.normal = f.unit_vector(np.asarray(normal))

        # the 'lower floor' is flat, represented by a circle
        self.circle_1 = Circle(self.center_point, self.radius_point, self.normal)
        self.radius = f.norm(self.radius_point - self.center_point)

        # geometry to project to
        import uuid
        geometry_name = f"sphere_{str(uuid.uuid1())[:8]}"
        self.geometry = {
            geometry_name: [
                'type sphere',
                f"origin ({self.center_point[0]} {self.center_point[1]} {self.center_point[2]})",
                f"radius {self.radius}",
            ]
        }

        # the 'upper floor' is obtained by revolving circle's points around an
        # axis lying in the circle plane (dome_rev_axis)
        def rotate_dome(point, angle):
            # point lies on circle plane; rotate it upwards, away from it
            radius_vector = f.unit_vector(point - self.center_point)
            dome_rev_axis = np.cross(radius_vector, self.normal)
    
            dome_point = f.arbitrary_rotation(
                point, dome_rev_axis, angle, self.center_point)

            return dome_point

        self.shell = [None]*12
        self.core = [None]*4

        # circle uses constants.circle_core_diagonal and circle_core_side
        # to calculate core-to-shell ratios;
        # analogously, hemisphere uses the angles for the former and the latter.
        side_angle = c.sphere_side_angle
        diagonal_angle = c.sphere_diagonal_angle

        # blocks around circumference
        for i, circle_face in enumerate(self.circle_1.shell_faces):
            points = circle_face.points

            start_angle = side_angle
            end_angle = diagonal_angle
            if i % 2 == 0:
                start_angle, end_angle = end_angle, start_angle

            inner_bottom_point_1 = points[0]
            outer_bottom_point_1 = points[1]

            inner_upper_point_1 = rotate_dome(inner_bottom_point_1, start_angle)
            outer_upper_point_1 = rotate_dome(outer_bottom_point_1, start_angle)

            inner_bottom_point_2 = points[3]
            outer_bottom_point_2 = points[2]

            inner_upper_point_2 = rotate_dome(inner_bottom_point_2, end_angle)
            outer_upper_point_2 = rotate_dome(outer_bottom_point_2, end_angle)

            lower_face = Face(
                [ # points
                    inner_bottom_point_1, outer_bottom_point_1,
                    outer_bottom_point_2, inner_bottom_point_2
                ],
                [ # edges: project rather than calculate points
                    None, geometry_name, None, None
                ]
            )

            upper_face = Face(
                [
                    inner_upper_point_1, outer_upper_point_1,
                    outer_upper_point_2, inner_upper_point_2
                ],
                [
                    None, geometry_name, None, None
                ]
            )

            loft = Loft(
                lower_face,
                upper_face,
                [None, geometry_name, geometry_name, None] # projected lofted edges
            )
            loft.block.project_face('right', geometry_name) # and projected block face
            self.shell[i] = loft

        # top and core blocks
        top_point = self.center_point + self.normal*self.radius
        mid_point = self.center_point + self.normal*self.radius*0.5

        for i in range(4):
            # two shell faces in a quarter of shell circle
            face_1 = self.shell[2*i].top_face
            face_2 = self.shell[(2*i)+1].top_face

            bottom_face = Face([
                mid_point, face_1.points[0], face_1.points[3], face_2.points[3]
            ])
            top_face = Face(
                [top_point, face_1.points[1], face_1.points[2], face_2.points[2]],
                [geometry_name]*4 # project all edges of this face
            )

            roof_loft = Loft(bottom_face, top_face)
            roof_loft.block.project_face('top', geometry_name)

            self.shell[8+i] = roof_loft

            core_loft = Loft(self.circle_1.core_faces[i], bottom_face)
            self.core[i] = core_loft

        self.operations = self.core + self.shell

    ### Patches
    def set_bottom_patch(self, patch_name):
        # only the 'bottom' shell blocks
        for s in self.shell[:8]:
            s.block.set_patch('bottom', patch_name)
        
        for c in self.core:
            c.block.set_patch('bottom', patch_name)

    def set_top_patch(self, patch_name):
        for s in self.shell[:8]:
            s.block.set_patch('right', patch_name)
        
        for s in self.shell[8:]:
            s.block.set_patch('top', patch_name)

    def set_outer_patch(self, patch_name):
        self.set_top_patch(patch_name)

    ### Chopping    
    def chop_tangential(self, **kwargs):
        # only chop outer shell blocks; 'roof' blocks are oriented differently
        for s in self.shell[:8]:
            s.chop(1, **kwargs)
            
    @classmethod
    def chain(cls, source, start_face=False):
        assert source.sketch_class == Circle

        # TODO: TEST
        if start_face:
            center_point = source.sketch_1.center_point
            radius_point = source.sketch_1.radius_point
            normal = -source.sketch_1.normal
        else:
            center_point = source.sketch_2.center_point
            radius_point = source.sketch_2.radius_point
            normal = source.sketch_2.normal

        return cls(center_point, radius_point, normal)