import numpy as np
import parameters as params
from regions.region import Region

import classy_blocks as cb
from classy_blocks.construct.edges import Line
from classy_blocks.construct.point import Point
from classy_blocks.util import functions as f


class InletPipe(Region):
    line_clamps = {20, 18, 16, 13, 12}
    plane_clamps = {0, 1, 8, 9, 5, 4, 11, 10}
    free_clamps = {3, 2, 6, 7}

    def __init__(self):
        self.inlet = self.create_inlet()

    def snap_point(self, point: Point) -> None:
        """Moves points on inlet's end face along x-axis
        so that they touch cyclone body"""
        # a.k.a. conform to that cylinder; those vertices will remain fixed
        y_pos = point.position[1]
        angle = np.arccos(y_pos / self.geo.r["body"])
        x_pos = -self.geo.r["body"] * np.sin(angle)
        point.translate([x_pos, 0, 0])

    def scale_point(self, point: Point) -> None:
        """Move a point radially towards axis of cyclone
        by a fraction of body thickness at this spot;
        creates a cone from which shell will be built"""
        vector = f.unit_vector(point.position)
        vector[2] = 0
        vector *= (self.geo.r["body"] - self.geo.r["pipe"]) * 0.3
        point.translate(-vector)

    def create_inlet(self):
        inlet = cb.SemiCylinder(
            self.geo.inlet[1], self.geo.inlet[2], self.geo.inlet[1] - f.vector(0, self.geo.r["inlet"], 0)
        )

        for op in inlet.operations:
            op.top_face.remove_edges()
            op.bottom_face.remove_edges()

            for point in op.top_face.points:
                # move points on inlet's end face so that they touch cyclone body,
                # a.k.a. conform to that cylinder; those vertices will remain fixed
                self.snap_point(point)

        for op in inlet.core:
            for point in op.top_face.points:
                self.scale_point(point)

        for op in inlet.shell:
            for i in (0, 3):
                self.scale_point(op.top_face.points[i])

        return inlet

    @property
    def inner_lofts(self):
        """Lofts that will be used to create inner ring, that is, those that
        have tops facing z-axis"""
        return self.inlet.core

    def chop(self):
        self.inlet.chop_axial(start_size=params.BULK_SIZE)

    @property
    def elements(self):
        return self.inlet.operations

    def project(self):
        for operation in self.inlet.shell:
            operation.project_side("right", "inlet", True, True)


class InletExtension(Region):
    def __init__(self, inlet: InletPipe):
        self.extension = cb.SemiCylinder(
            self.geo.inlet[0], self.geo.inlet[1], self.geo.inlet[0] - f.vector(0, self.geo.r["inlet"], 0)
        )

        for i, operation in enumerate(self.extension.operations):
            operation.top_face = inlet.elements[i].bottom_face

    def chop(self):
        self.extension.chop_axial(start_size=params.BULK_SIZE)

    @property
    def elements(self):
        return self.extension.operations

    def set_patches(self):
        self.extension.set_start_patch("inlet")
