"""A copy of py_gear_gen (https://github.com/heartworm/py_gear_gen),
slightly modified to fit in classy_blocks for the gear pump example"""

import numpy as np

from classy_blocks.util import functions as f


class DimensionError(Exception):
    pass


def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def flip_matrix(h, v):
    return [[-1 if h else 1, 0], [0, -1 if v else 1]]


def polar_to_cart(*coords):
    if len(coords) == 1:
        coords = coords[0]
    r, ang = coords
    return r * np.cos(ang), r * np.sin(ang)


def cart_to_polar(*coords):
    if len(coords) == 1:
        coords = coords[0]
    x, y = coords
    return np.sqrt(x * x + y * y), np.arctan2(y, x)


class InvoluteGear:
    def __init__(
        self,
        module=1,
        teeth=30,
        pressure_angle_deg=20.0,
        fillet=0.0,
        backlash=0.0,
        max_steps=100,
        arc_step_size=0.1,
        reduction_tolerance_deg=0.0,
        dedendum_factor=1.157,
        addendum_factor=1.0,
        ring=False,
    ):
        """
        Construct an involute gear, ready for generation using one of the generation methods.
        :param module: The 'module' of the gear. (Diameter / Teeth)
        :param teeth: How many teeth in the desired gear.
        :param pressure_angle_deg: The pressure angle of the gear in DEGREES.
        :param fillet: The radius of the fillet connecting a tooth to the root circle. NOT WORKING in ring gear.
        :param backlash: The circumfrential play between teeth,
                         if meshed with another gear of the same backlash held stationary
        :param max_steps: Maximum steps allowed to generate the involute profile. Higher is more accurate.
        :param arc_step_size: The step size used for generating arcs.
        :param ring: True if this is a ring (internal) gear, otherwise False.
        """

        pressure_angle = f.deg2rad(pressure_angle_deg)
        self.reduction_tolerance = f.deg2rad(reduction_tolerance_deg)
        self.module = module
        self.teeth = teeth
        self.pressure_angle = pressure_angle

        # Addendum is the height above the pitch circle that the tooth extends to
        self.addendum = addendum_factor * module
        # Dedendum is the depth below the pitch circle the root extends to. 1.157 is a std value allowing for clearance.
        self.dedendum = dedendum_factor * module

        # If the gear is a ring gear, then the clearance needs to be on the other side
        if ring:
            temp = self.addendum
            self.addendum = self.dedendum
            self.dedendum = temp

        # The radius of the pitch circle
        self.pitch_radius = (module * teeth) / 2
        # The radius of the base circle, used to generate the involute curve
        self.base_radius = np.cos(pressure_angle) * self.pitch_radius
        # The radius of the gear's extremities
        self.outer_radius = self.pitch_radius + self.addendum
        # The radius of the gaps between the teeth
        self.root_radius = self.pitch_radius - self.dedendum

        # The radius of the fillet circle connecting the tooth to the root circle
        self.fillet_radius = fillet if not ring else 0

        # The angular width of a tooth and a gap. 360 degrees divided by the number of teeth
        self.theta_tooth_and_gap = np.pi * 2 / teeth
        # Converting the circumfrential backlash into an angle
        angular_backlash = backlash / 2 / self.pitch_radius
        # The angular width of the tooth at the pitch circle minus backlash, not taking the involute into account
        self.theta_tooth = self.theta_tooth_and_gap / 2 + (-angular_backlash if not ring else angular_backlash)
        # Where the involute profile intersects the pitch circle, found on iteration.
        self.theta_pitch_intersect = None
        # The angular width of the full tooth, at the root circle
        self.theta_full_tooth = 0

        self.max_steps = max_steps
        self.arc_step_size = arc_step_size

    """
    Reduces a line of many points to less points depending on the allowed angle tolerance
    """

    def reduce_polyline(self, polyline):
        vertices = [[], []]
        last_vertex = [polyline[0][0], polyline[1][0]]

        # Look through all vertices except start and end vertex
        # Calculate by how much the lines before and after the vertex
        # deviate from a straight path.
        # If the deviation angle exceeds the specification, store it
        for vertex_idx in range(1, len(polyline[0]) - 1):
            next_slope = np.arctan2(
                polyline[1][vertex_idx + 1] - polyline[1][vertex_idx + 0],
                polyline[0][vertex_idx + 1] - polyline[0][vertex_idx + 0],
            )
            prev_slope = np.arctan2(
                polyline[1][vertex_idx - 0] - last_vertex[1], polyline[0][vertex_idx - 0] - last_vertex[0]
            )

            deviation_angle = abs(prev_slope - next_slope)

            if deviation_angle > self.reduction_tolerance:
                vertices[0] += [polyline[0][vertex_idx]]
                vertices[1] += [polyline[1][vertex_idx]]
                last_vertex = [polyline[0][vertex_idx], polyline[1][vertex_idx]]

        # Return vertices along with first and last point of the original polyline
        return np.array(
            [
                np.concatenate([[polyline[0][0]], vertices[0], [polyline[0][-1]]]),
                np.concatenate([[polyline[1][0]], vertices[1], [polyline[1][-1]]]),
            ]
        )

    def generate_half_tooth(self):
        """
        Generate half an involute profile, ready to be mirrored in order to create one symmetrical involute tooth
        :return: A numpy array, of the format [[x1, x2, ... , xn], [y1, y2, ... , yn]]
        """
        # Theta is the angle around the circle, however PHI is simply a parameter for iteratively building the involute
        phis = np.linspace(0, np.pi, self.max_steps)
        points = []
        reached_limit = False
        self.theta_pitch_intersect = None

        for phi in phis:
            x = (self.base_radius * np.cos(phi)) + (phi * self.base_radius * np.sin(phi))
            y = (self.base_radius * np.sin(phi)) - (phi * self.base_radius * np.cos(phi))
            point = (x, y)
            dist, theta = cart_to_polar(point)

            if self.theta_pitch_intersect is None and dist >= self.pitch_radius:
                self.theta_pitch_intersect = theta
                self.theta_full_tooth = self.theta_pitch_intersect * 2 + self.theta_tooth
            elif self.theta_pitch_intersect is not None and theta >= self.theta_full_tooth / 2:
                reached_limit = True
                break

            if dist >= self.outer_radius:
                points.append(polar_to_cart((self.outer_radius, theta)))
            elif dist <= self.root_radius:
                points.append(polar_to_cart((self.root_radius, theta)))
            else:
                points.append((x, y))

        if not reached_limit:
            raise Exception("Couldn't complete tooth profile.")

        return np.transpose(points)

    def generate_half_root(self):
        """
        Generate half of the gap between teeth, for the first tooth
        :return: A numpy array, of the format [[x1, x2, ... , xn], [y1, y2, ... , yn]]
        """
        root_arc_length = (self.theta_tooth_and_gap - self.theta_full_tooth) * self.root_radius

        points_root = []
        for theta in np.arange(
            self.theta_full_tooth,
            self.theta_tooth_and_gap / 2 + self.theta_full_tooth / 2,
            self.arc_step_size / self.root_radius,
        ):
            # The current circumfrential position we are in the root arc, starting from 0
            arc_position = (theta - self.theta_full_tooth) * self.root_radius
            # If we are in the extemities of the root arc (defined by fillet_radius), then we are in a fillet
            in_fillet = min((root_arc_length - arc_position), arc_position) < self.fillet_radius

            r = self.root_radius

            if in_fillet:
                # Add a circular profile onto the normal root radius to form the fillet.
                # High near the edges, small towards the centre
                # The min() function handles the situation where the fillet size is massive and overlaps itself
                circle_pos = min(arc_position, (root_arc_length - arc_position))
                r = r + (
                    self.fillet_radius - np.sqrt(pow(self.fillet_radius, 2) - pow(self.fillet_radius - circle_pos, 2))
                )
            points_root.append(polar_to_cart((r, theta)))

        return np.transpose(points_root)

    def generate_roots(self):
        """
        Generate both roots on either side of the first tooth
        :return: A numpy array, of the format
                 [ [[x01, x02, ... , x0n], [y01, y02, ... , y0n]], [[x11, x12, ... , x1n], [y11, y12, ... , y1n]] ]
        """
        self.half_root = self.generate_half_root()
        self.half_root = np.dot(rotation_matrix(-self.theta_full_tooth / 2), self.half_root)
        points_second_half = np.dot(flip_matrix(False, True), self.half_root)
        points_second_half = np.flip(points_second_half, 1)
        self.roots = [points_second_half, self.half_root]

        # Generate a second set of point-reduced root
        self.half_root_reduced = self.reduce_polyline(self.half_root)
        points_second_half = np.dot(flip_matrix(False, True), self.half_root_reduced)
        points_second_half = np.flip(points_second_half, 1)
        self.roots_reduced = [points_second_half, self.half_root_reduced]

        return self.roots_reduced

    def generate_tooth(self):
        """
        Generate only one involute tooth, without an accompanying tooth gap
        :return: A numpy array, of the format [[x1, x2, ... , xn], [y1, y2, ... , yn]]
        """
        self.half_tooth = self.generate_half_tooth()
        self.half_tooth = np.dot(rotation_matrix(-self.theta_full_tooth / 2), self.half_tooth)
        points_second_half = np.dot(flip_matrix(False, True), self.half_tooth)
        points_second_half = np.flip(points_second_half, 1)
        self.tooth = np.concatenate((self.half_tooth, points_second_half), axis=1)

        # Generate a second set of point-reduced teeth
        self.half_tooth_reduced = self.reduce_polyline(self.half_tooth)
        points_second_half = np.dot(flip_matrix(False, True), self.half_tooth_reduced)
        points_second_half = np.flip(points_second_half, 1)
        self.tooth_reduced = np.concatenate((self.half_tooth_reduced, points_second_half), axis=1)

        return self.tooth_reduced

    def generate_tooth_and_gap(self):
        """
        Generate only one tooth and one root profile, ready to be duplicated by rotating around the gear center
        :return: A numpy array, of the format [[x1, x2, ... , xn], [y1, y2, ... , yn]]
        """

        points_tooth = self.generate_tooth()
        points_roots = self.generate_roots()
        self.tooth_and_gap = np.concatenate((points_roots[0], points_tooth, points_roots[1]), axis=1)
        return self.tooth_and_gap

    def generate_gear(self):
        """
        Generate the gear profile, and return a sequence of co-ordinates representing the outline of the gear
        :return: A numpy array, of the format [[x1, x2, ... , xn], [y1, y2, ... , yn]]
        """

        points_tooth_and_gap = self.generate_tooth_and_gap()
        points_teeth = [
            np.dot(rotation_matrix(self.theta_tooth_and_gap * n), points_tooth_and_gap) for n in range(self.teeth)
        ]
        points_gear = np.concatenate(points_teeth, axis=1)
        return points_gear

    def get_point_list(self):
        """
        Generate the gear profile, and return a sequence of co-ordinates representing the outline of the gear
        :return: A numpy array, of the format [[x1, y2], [x2, y2], ... , [xn, yn]]
        """

        gear = self.generate_gear()
        return np.transpose(gear)
