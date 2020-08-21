import numpy as np

from classes.mesh import Mesh
from classes.block import Block

from operations.base import Face
from operations.operations import Loft, Revolve

from shapes.circle import Circle

from util.methematics import functions as g
from util import constants

class Elbow:
    def __init__(self, center_point_1:list, radius_point_1:list, normal_1:list,
        sweep_angle:float, arc_center:list, rotation_axis:list, radius_2:float):
        super().__init__()

        self.center_point_1 = np.array(center_point_1)
        self.radius_point_1 = np.array(radius_point_1)
        self.normal_1 = np.array(normal_1)

        self.sweep_angle = sweep_angle
        self.arc_center = np.array(arc_center)
        self.rotation_axis = np.array(rotation_axis)

        self.radius_1_length = g.norm(self.center_point_1 - self.radius_point_1)
        self.radius_2_length = radius_2

        self.circle_1 = Circle(self.center_point_1, self.radius_point_1, self.normal_1)

        def rotate_circle(angle, radius):
            center_point = g.arbitrary_rotation(self.center_point_1, self.rotation_axis, angle, self.arc_center)

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
        return [self.core] + self.shell
    
    @property
    def blocks(self):
        return [o.block for o in self.operations]
    
    def set_patch(self, side, patch_name):
        for b in self.blocks:
            b.set_patch(side, patch_name)

    def set_bottom_patch(self, patch_name):
        self.set_patch('bottom', patch_name)

    def set_top_patch(self, patch_name):
        self.set_patch('top', patch_name)
    
    def set_outer_patch(self, patch_name):
        for s in self.shell:
            s.block.set_patch('right', patch_name)

    def set_axial_cell_count(self, count):
        for o in self.operations:
            o.set_cell_count(2, count)

    def set_radial_cell_count(self, count):
        for s in self.shell:
            s.set_cell_count(0, count)

    def set_tangential_cell_count(self, count):
        for s in self.shell:
            s.set_cell_count(1, count)

        self.core.set_cell_count(0, count)
        self.core.set_cell_count(1, count)

    def set_axial_cell_size(self, size):
        # 'core' is the first block to be added;
        # other blocks' grading will be copied from it
        self.core.grade_to_size(2, size)

    def set_outer_cell_size(self, size):
        # only set grading for the first shell block,
        # mesh will copy it to others
        self.shell[0].grade_to_size(0, -size)


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
        self.radius_point_2 = self.axis_point_2 + g.unit_vector(self.radius_1)*self.radius_2_length

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
        radius_1 = g.norm(np.array(radius_point_1) - np.array(axis_point_1))

        super().__init__(axis_point_1, axis_point_2, radius_point_1, radius_1)

class Ring:
    def __init__(self, axis_point_1:list, axis_point_2:list, face:Face, n_blocks=4):
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

    @property
    def operations(self):
        return self.shell
    
    @property
    def blocks(self):
        return [o.block for o in self.operations]
    
    def set_axial_cell_count(self, count):
        for o in self.operations:
            o.set_cell_count(0, count)

    def set_radial_cell_count(self, count):
        for o in self.operations:
            o.set_cell_count(1, count)

    def set_tangential_cell_count(self, count):
        for o in self.operations:
            o.set_cell_count(2, count)
    
    def set_axial_cell_size(self, size):
        for o in self.operations:
            o.grade_to_size(0, size)
    
    def set_radial_cell_size(self, size):
        for o in self.operations:
            o.grade_to_size(1, size)

    def set_patch(self, side, patch_name):
        for b in self.blocks:
            b.set_patch(side, patch_name)

    def set_bottom_patch(self, patch_name):
        self.set_patch('left', patch_name)

    def set_top_patch(self, patch_name):
        self.set_patch('right', patch_name)
    
    def set_outer_patch(self, patch_name):
        for s in self.shell:
            s.block.set_patch('back', patch_name)

    def set_inner_patch(self, patch_name):
        for s in self.shell:
            s.block.set_patch('front', patch_name)