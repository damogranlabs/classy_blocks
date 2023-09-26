import dataclasses
from typing import List

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.items.side import Side
from classy_blocks.items.vertex import Vertex
from classy_blocks.types import OrientType
from classy_blocks.util.constants import SIDES_MAP


@dataclasses.dataclass
class ProjectedFace:
    """An entry in blockMeshDict.faces"""

    side: Side
    label: str

    def __eq__(self, other):
        return self.side == other.side

    @property
    def description(self) -> str:
        """Formats the 'faces' list to be output into blockMeshDict"""
        return f"\tproject {self.side.description} {self.label}\n"


class FaceList:
    """Handling of projected faces (the 'faces' part of blockMeshDict)"""

    def __init__(self) -> None:
        self.faces: List[ProjectedFace] = []

    def find_existing(self, side: Side) -> bool:
        """Returns true if side in arguments exists already"""
        for face in self.faces:
            if face.side == side:
                return True

        return False

    def add(self, vertices: List[Vertex], operation: Operation) -> None:
        """Collect projected sides from operation data"""

        for index, orient in enumerate(SIDES_MAP):
            label = operation.side_projects[index]

            if label is not None:
                self.add_side(Side(orient, vertices), label)

        # don't forget bottom and top faces
        self.add_face(vertices, "bottom", operation.bottom_face)
        self.add_face(vertices, "top", operation.top_face)

    def add_face(self, vertices: List[Vertex], orient: OrientType, face: Face) -> None:
        """Add a face to faces list (if it is projected to anything)"""
        if face.projected_to is not None:
            self.add_side(Side(orient, vertices), face.projected_to)

    def add_side(self, side: Side, label: str) -> None:
        """Adds a projected face (side) to the list if it's not there yet"""
        if not self.find_existing(side):
            self.faces.append(ProjectedFace(side, label))

    def clear(self) -> None:
        """Removes collected faces"""
        self.faces.clear()

    @property
    def description(self) -> str:
        """Formats the 'faces' list to be output into blockMeshDict"""
        out = "faces\n(\n"

        for face in self.faces:
            out += face.description

        out += ");\n\n"

        return out
