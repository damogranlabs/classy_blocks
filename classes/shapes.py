import numpy as np

from ..classes.operations import Face, Loft, Revolve, Extrude

from ..util import functions as f
from ..util import constants

class Box(Extrude):
    def __init__(self, point_min:list, point_max:list):
        """ A box, aligned with coordinate system """
        base = Face([
            [point_min[0], point_min[1], point_min[2]],
            [point_max[0], point_min[1], point_min[2]],
            [point_max[0], point_max[1], point_min[2]],
            [point_min[0], point_max[1], point_min[2]],
        ], check_coplanar=True)

        extrude_vector = [0, 0, point_max[2] - point_min[2]]

        super().__init__(base, extrude_vector)

class Circle:
    """ A base shape for Circle, Annulus and whatnot """
    def __init__(self, center_point, radius_point, normal):
        self.center_point = np.array(center_point)
        self.radius_point = np.array(radius_point)
        self.normal = np.array(normal)

        self.radius_vector = self.radius_point - self.center_point
        self.radius = f.norm(self.radius_vector)

        vertex_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
        edge_angles = np.linspace(np.pi/4, 2*np.pi + np.pi/4, 4, endpoint=False)

        # create faces
        # core
        vertex_ratio = self.get_vertex_ratio()
        core_point = self.center_point + self.radius_vector*vertex_ratio
        core_points = [self._rotate(core_point, a) for a in vertex_angles]

        edge_ratio = self.get_edge_ratio()
        core_edge = self.center_point + self.radius_vector*edge_ratio
        core_edges = [self._rotate(core_edge, a) for a in edge_angles]

        self.core_face = Face(core_points, core_edges)

        # 4 shell faces around core
        shell_face_points = [core_points[0], self.radius_point, self._rotate(self.radius_point, np.pi/2), core_points[1]]
        shell_edge_points = [None, self._rotate(self.radius_point, np.pi/4), None, None]
        shell_face = Face(shell_face_points, shell_edge_points)

        self.shell_faces = [
            shell_face.rotate(self.normal, a, origin=self.center_point) for a in vertex_angles
        ]

    def get_vertex_ratio(self):
        # ready-to-be-overridden
        return constants.frustum_core_to_outer
    
    def get_edge_ratio(self):
        # ready-to-be-overridden
        return constants.frustum_edge_to_outer

    def _rotate(self, p, angle):
        # just a shortcut for maths in __init__
        return f.arbitrary_rotation(p, self.normal, angle, self.center_point)

    def translate(self, vector):
        vector = np.array(vector)
        
        center_point = self.center_point + vector
        radius_point = self.radius_point + vector
        
        # normal does not change during translation
        return self.__class__(center_point, radius_point, self.normal)

    def rotate(self, axis, angle, origin):
        # rotate center point and radius point around origin;
        # rotate normal around zero
        new_center_point = f.arbitrary_rotation(self.center_point, axis, angle, origin)
        new_radius_point = f.arbitrary_rotation(self.radius_point, axis, angle, origin)
        new_normal = f.arbitrary_rotation(self.normal, axis, angle, [0, 0, 0])

        return self.__class__(new_center_point, new_radius_point, new_normal)

    def scale(self, new_radius):
        radius_vector = self.radius_vector/f.norm(self.radius_vector) * new_radius
        new_radius_point = self.center_point + radius_vector

        # normal does not change during scaling
        return self.__class__(self.center_point, new_radius_point, self.normal)


class Elbow:
    """ A general round shape """

    # mappings between block sides and 'human-readable' geometry
    top_patch = 'top'
    bottom_patch = 'bottom'
    outer_patch = 'right'

    axial_axis = 2
    radial_axis = 0
    tangential_axis = 1

    def __init__(self, center_point_1:list, radius_point_1:list, normal_1:list,
        sweep_angle:float, arc_center:list, rotation_axis:list, radius_2:float):
        super().__init__()

        self.center_point_1 = np.array(center_point_1)
        self.radius_point_1 = np.array(radius_point_1)
        self.normal_1 = np.array(normal_1)

        self.sweep_angle = sweep_angle
        self.arc_center = np.array(arc_center)
        self.rotation_axis = np.array(rotation_axis)

        self.radius_1_length = f.norm(self.center_point_1 - self.radius_point_1)
        self.radius_2_length = radius_2

        self.circle_1 = Circle(self.center_point_1, self.radius_point_1, self.normal_1)

        def rotate_circle(angle, radius):
            center_point = f.arbitrary_rotation(self.center_point_1, self.rotation_axis, angle, self.arc_center)

            return self.circle_1 \
                .rotate(self.rotation_axis, angle, self.arc_center) \
                .translate(center_point - self.center_point_1) \
                .scale(radius)

        self.circle_2 = rotate_circle(self.sweep_angle, self.radius_2_length)

        # a circle in between with radius between the two
        self.circle_between = rotate_circle(self.sweep_angle/2, (self.radius_2_length + self.radius_1_length)/2)

        self.core = Loft(
            self.circle_1.core_face,
            self.circle_2.core_face,
            self.circle_between.core_face.points)

        # shell: loft again 4 times but with only 1 curved edge
        self.shell = [
            Loft(self.circle_1.shell_faces[i],self.circle_2.shell_faces[i],self.circle_between.shell_faces[i].points)
            for i in range(4)
        ]
    
    @property
    def operations(self):
        if hasattr(self, 'core'):
            return [self.core] + self.shell
        
        return self.shell
    
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

    ### Counts and gradings
    def chop_axial(self, **kwargs):
        self.shell[0].chop(self.axial_axis, **kwargs)

    def chop_radial(self, **kwargs):
        # set cell count on the outside
        kwargs['invert'] = not kwargs.get('invert')

        for s in self.shell:
            s.chop(self.radial_axis, **kwargs)

    def chop_tangential(self, **kwargs):
        for s in self.shell:
            s.chop(self.tangential_axis, **kwargs)

    def set_cell_zone(self, cell_zone):
        for b in self.blocks:
            b.cell_zone = cell_zone

class Frustum(Elbow):
    def __init__(self, axis_point_1:list, axis_point_2:list, radius_point_1:list, radius_2:float):
        """ Creates a cone frustum (truncated cylinder) with axis between points
        'axis_point_1' and 'axis_point_2'. There's one block in the center and 4 around it.
        'radius_point_1' define starting point for blocks and radius_2 defines end radius
        (NOT A POINT! 'radius_point_2' is calculated from the other 3 points so that
        all four lie on the same plane). """
        self.axis_point_1 = np.array(axis_point_1)
        self.radius_point_1 = np.array(radius_point_1)
        self.radius_1 = self.radius_point_1 - self.axis_point_1

        self.axis_point_2 = np.array(axis_point_2)
        self.radius_2_length = radius_2
        self.radius_point_2 = self.axis_point_2 + f.unit_vector(self.radius_1)*self.radius_2_length

        self.axis = self.axis_point_2 - self.axis_point_1

        self.circle_1 = Circle(self.axis_point_1, self.radius_point_1, self.axis)
        self.circle_2 = self.circle_1.translate(self.axis).scale(self.radius_2_length)

        self.core = Loft(self.circle_1.core_face, self.circle_2.core_face)

        # shell: loft again 4 times but with only 1 curved edge
        self.shell = [
            Loft(self.circle_1.shell_faces[i], self.circle_2.shell_faces[i])
            for i in range(4)
        ]

class Cylinder(Frustum):
    def __init__(self, axis_point_1:list, axis_point_2:list, radius_point_1:list):
        """ a Frustum with constant radius """
        radius_1 = f.norm(np.array(radius_point_1) - np.array(axis_point_1))

        super().__init__(axis_point_1, axis_point_2, radius_point_1, radius_1)

class RevolvedRing(Elbow):
    axial_axis = 0
    radial_axis = 1
    tangential_axis = 2

    bottom_patch = 'left'
    top_patch = 'right'
    outer_patch = 'back'

    def __init__(self, axis_point_1:list, axis_point_2:list, face:Face, n_blocks=4):
        """ a ring created by revolving its cross-section around given axis """
        # create a revolve from face around axis
        self.axis_point_1 = np.array(axis_point_1)
        self.axis_point_2 = np.array(axis_point_2)
        self.face = face
        self.n_blocks = n_blocks

        revolve_angle = 2*np.pi/n_blocks
        axis = self.axis_point_2 - self.axis_point_1
        revolve = Revolve(self.face, revolve_angle, axis, self.axis_point_1)

        revolve_angles = np.linspace(0, 2*np.pi, self.n_blocks, endpoint=False)

        self.shell = [revolve.rotate(axis, a, self.axis_point_1) for a in revolve_angles]

    def chop_radial(self, **kwargs):
        # set cell count on the outside
        # different orientation than Elbow
        kwargs['invert'] = not kwargs.get('invert')
        super().chop_radial(**kwargs)

    def set_inner_patch(self, patch_name):
        for s in self.shell:
            s.block.set_patch('front', patch_name)

class ExtrudedRing(RevolvedRing):
    """ a ring specified like a Cylinder """
    def __init__(self, axis_point_1:list, axis_point_2:list,
        inner_radius_point_1:list, outer_radius:float, n_blocks=4):

        # calculate parameters for
        axis_point_1 = np.array(axis_point_1)
        axis_point_2 = np.array(axis_point_2)
        inner_radius_point_1 = np.array(inner_radius_point_1)

        inner_radius_vector = inner_radius_point_1 - axis_point_1
        inner_radius_point_2 = axis_point_2 + inner_radius_vector

        outer_radius_vector = f.unit_vector(inner_radius_vector)*outer_radius
        outer_radius_point_1 = axis_point_1 + outer_radius_vector
        outer_radius_point_2 = axis_point_2 + outer_radius_vector

        # prepare everything for a RevolvedRing, because laziness
        revolve_face = Face(
            [
                inner_radius_point_1,
                inner_radius_point_2,
                outer_radius_point_2,
                outer_radius_point_1
            ]
        )
        
        super().__init__(axis_point_1, axis_point_2, revolve_face, n_blocks)
