import abc
from typing import ClassVar

from classy_blocks.cbtyping import IndexType, OrientType
from classy_blocks.optimize.cell import CellBase, HexCell, QuadCell


class FaceRegistryBase(abc.ABC):
    cell_type: ClassVar[type[CellBase]]

    def get_key_from_side(self, cell: int, side: OrientType) -> tuple:
        cell_indexes = self.addressing[cell]
        face_corners = self.orient_indexes[side]
        face_indexes = [cell_indexes[c] for c in face_corners]

        return tuple(sorted(face_indexes))

    def __init__(self, addressing: list[IndexType]):
        self.addressing = addressing

        # this is almost equal to constants.FACE_MAP except
        # faces are oriented so that they point towards cell centers
        self.orient_indexes: dict[OrientType, IndexType] = {
            name: self.cell_type.side_indexes[i] for i, name in enumerate(self.cell_type.side_names)
        }

        # will hold faces, accessible in O(1) time, and their coincident cell(s);
        # boundary faces will have a single coincident cell, internal two
        self.faces: dict[tuple, set[int]] = {}

        for cell in range(len(self.addressing)):
            for side in self.orient_indexes.keys():
                face_key = self.get_key_from_side(cell, side)

                if face_key not in self.faces:
                    self.faces[face_key] = set()

                self.faces[face_key].add(cell)

    def get_cells(self, cell: int, side: OrientType) -> set[int]:
        return self.faces[self.get_key_from_side(cell, side)]


class QuadFaceRegistry(FaceRegistryBase):
    cell_type = QuadCell


class HexFaceRegistry(FaceRegistryBase):
    cell_type = HexCell
