from typing import List

import numpy as np

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.modify.reorient.viewpoint import ViewpointReorienter
from classy_blocks.util import functions as f


class FacePair:
    def __init__(self, face_1: Face, face_2: Face):
        self.face_1 = face_1
        self.face_2 = face_2

    @property
    def distance(self) -> float:
        """Returns distance between two faces' centers"""
        return f.norm(self.face_1.center - self.face_2.center)

    @property
    def alignment(self) -> float:
        """Returns a scalar number that is a measure of how well the
        two faces are aligned, a.k.a. how well their normals align"""
        vconn = f.unit_vector(self.face_2.center - self.face_1.center)
        return np.dot(vconn, self.face_1.normal) ** 3 + np.dot(-vconn, self.face_2.normal) ** 3


class Connector(Operation):
    """A normal Loft but automatically finds and reorders appropriate faces between
    two arbitrary given blocks.

    The recipe is as follows:
      1. Find a pair of faces whose normals are most nicely aligned
      2. Create a loft that connects them
      3. Reorder the loft so that is is properly oriented

    The following limitations apply:
    "Closest faces" might be an ill-defined term; for example,
    imagine two boxes:
          ___
         | 2 |
         |___|
     ___
    | 1 |
    |___|

    Here, multiple different faces can be found.

    Reordering relies on ViewpointReorienter; see the documentation on that
    for its limitations.

    Resulting loft will have the bottom face coincident with operation_1
    and top face with operation_2.
    Axis 2 is always between the two operations but axes 0 and 1
    depend on positions of operations and is not exactly defined.
    To somewhat alleviate this confusion it is
    recommended to chop operation 1 or 2 in axes 0 and 1 and
    only provide chopping for axis 2 of connector."""

    def __init__(self, operation_1: Operation, operation_2: Operation):
        self.operation_1 = operation_1
        self.operation_2 = operation_2

        all_pairs: List[FacePair] = []
        for orient_1, face_1 in operation_1.get_all_faces().items():
            if orient_1 in ("bottom", "left", "front"):
                face_1.invert()
            for orient_2, face_2 in operation_2.get_all_faces().items():
                if orient_2 in ("bottom", "left", "front"):
                    face_2.invert()
                all_pairs.append(FacePair(face_1, face_2))

        all_pairs.sort(key=lambda pair: pair.distance)
        all_pairs = all_pairs[:9]
        all_pairs.sort(key=lambda pair: pair.alignment)

        start_face = all_pairs[-1].face_1
        end_face = all_pairs[-1].face_2

        super().__init__(start_face, end_face)

        viewpoint = operation_1.center + 2 * (operation_1.top_face.center - operation_1.bottom_face.center)
        ceiling = operation_1.center + 2 * (operation_2.center - operation_1.center)
        reorienter = ViewpointReorienter(viewpoint, ceiling)
        reorienter.reorient(self)
