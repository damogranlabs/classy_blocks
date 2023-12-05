import numpy as np

from classy_blocks.construct.operations.operation import Operation
from classy_blocks.modify.reorient.viewpoint import ViewpointReorienter
from classy_blocks.types import NPPointType
from classy_blocks.util import functions as f
from classy_blocks.util.constants import FACE_MAP


class Connector(Operation):
    """A normal Loft but automatically finds and reorders appropriate faces between
    two arbitrary given blocks.

    The recipe is as follows:
      1. Find a pair of faces that are closest together
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

    @staticmethod
    def _find_closest_face(point: NPPointType, op: Operation):
        """Finds a face in operation that is closest to a given point"""
        faces = [op.get_face(side) for side in FACE_MAP.keys()]
        distances = [f.norm(face.center - point) for face in faces]
        return faces[np.argmin(distances)]

    def __init__(self, operation_1: Operation, operation_2: Operation):
        self.operation_1 = operation_1
        self.operation_2 = operation_2

        start_face = self._find_closest_face(self.operation_2.center, operation_1)
        end_face = self._find_closest_face(start_face.center, self.operation_2)
        start_face = self._find_closest_face(end_face.center, self.operation_1)

        super().__init__(start_face, end_face)

        viewpoint = self.operation_1.center + 10 * (
            self.operation_1.top_face.center - self.operation_1.bottom_face.center
        )
        ceiling = self.operation_1.center + 10 * (self.operation_2.center - self.operation_1.center)
        reorienter = ViewpointReorienter(viewpoint, ceiling)
        reorienter.reorient(self)
