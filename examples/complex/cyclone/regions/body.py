from typing import List, Sequence

import parameters as params
from geometry import geometry as geo
from regions.region import Region

import classy_blocks as cb
from classy_blocks.base.transforms import Scaling, Translation
from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.construct.shape import LoftedShape
from classy_blocks.util import functions as f
from classy_blocks.util.constants import TOL


class ChainSketch(Sketch):
    """Collects bottom-most faces of given regions and creates a sketch from them"""

    @staticmethod
    def _get_faces(ops: Sequence[Operation]) -> List[cb.Face]:
        """Skims faces from given operations and reorders them so
        that they are ready for extruding or whatever"""
        faces: List[cb.Face] = []

        for operation in ops:
            # TODO: copy edges and other data too
            face = operation.get_closest_face([0, 0, 10 * geo.z["cone"]])

            # skip faces, made from inlet core blocks
            if face.center[2] > 0.99 * (geo.z["skirt"]):
                continue

            # reorient
            if face.normal[2] > 0:
                face.invert()

            faces.append(face)

        return faces

    @staticmethod
    def _add_arcs(faces: List[cb.Face]) -> None:
        """Adds arc edges between outermost points of outermost faces
        to create round shape"""
        # add arcs to outermost radius (body and cone)
        max_radius = 0
        for face in faces:
            radii = [f.to_polar(point, axis="z")[0] for point in face.point_array]
            max_radius = max(max_radius, max(radii))

        for face in faces:
            for i in range(4):
                polar_1 = f.to_polar(face.point_array[i], axis="z")
                polar_2 = f.to_polar(face.point_array[(i + 1) % 4], axis="z")

                if abs(polar_1[0] - max_radius) < 2 * TOL and abs(polar_2[0] - max_radius) < 2 * TOL:
                    face.add_edge(i, cb.Origin([0, 0, polar_1[2]]))

    def __init__(self, regions: List[Region]):
        # gather faces
        ops: Sequence[Operation] = []
        for region in regions:
            ops += region.elements

        faces = self._get_faces(ops)
        self._add_arcs(faces)

        self._faces = faces

    @property
    def faces(self):
        return self._faces

    @property
    def center(self):
        return f.vector(0, 0, self.faces[0].center[2])

    @property
    def n_segments(self):
        return 12


class BodyShape(LoftedShape):
    def chop_axial(self, **kwargs):
        self.operations[0].chop(2, **kwargs)

    def chop_radial(self, **kwargs):
        pass

    def chop_tangential(self, **kwargs):
        pass

    @property
    def shell(self):
        return []

    @property
    def core(self):
        return []


class UpperBody(Region):
    def __init__(self, sketch: ChainSketch, length: float):
        self.sketch = sketch
        self.body = BodyShape(self.sketch, [Translation([0, 0, -length])])

    def chop(self):
        self.elements[0].chop(2, start_size=params.BULK_SIZE)

    @property
    def elements(self):
        return self.body.operations


class LowerBody(UpperBody):
    pass


class Cone(UpperBody):
    def __init__(self, sketch: ChainSketch):
        self.sketch = sketch

        self.body = BodyShape(
            self.sketch,
            [
                Translation([0, 0, -self.geo.l["cone"]]),
                Scaling(self.geo.r["cone"] / self.geo.r["body"], f.vector(0, 0, self.geo.z["cone"])),
            ],
        )

    def chop(self):
        self.elements[0].chop(2, start_size=params.BULK_SIZE)
