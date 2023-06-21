from typing import ClassVar, Dict, Generic, List, Optional, Set, Tuple, TypeVar

from classy_blocks.types import AxisType
from classy_blocks.util import constants

BeamT = TypeVar("BeamT")


class Frame(Generic[BeamT]):
    """A two-dimensional dictionary for holding data
    between each end of hexahedra edges (called 'beam' generically
    like a cuve's frame would be created from those);

    An edge/wire/whatever object between vertices 0 and 1
    can be accessed as frame[0][1]. Diagonals are not available.

    Arguments only provide classes for type hinting:
    - beam_class: a class that is associated with a pair of corners

    Numbering and axes reflect that of block definition.
    After the Frame is created, entities must be added separately
    with appropriate methods."""

    valid_pairs: ClassVar[List[Set[int]]] = [set(pair) for pair in constants.EDGE_PAIRS]

    def __init__(self) -> None:
        self.beams: List[Dict[int, Optional[BeamT]]] = [{} for _ in range(8)]

        # create wires and connections for quicker addressing
        for axis in (0, 1, 2):
            for pair in constants.AXIS_PAIRS[axis]:
                self.add_beam(pair[0], pair[1], None)

    def add_beam(self, corner_1: int, corner_2: int, beam: Optional[BeamT]) -> None:
        """Adds an element between given corners;
        raises an exception if the given pair does not represent a beam"""
        if {corner_1, corner_2} not in self.valid_pairs:
            raise ValueError(
                f"Invalid combination of corners. Valid pairs: {self.valid_pairs}, got: {corner_1, corner_2}"
            )

        self.beams[corner_1][corner_2] = beam
        self.beams[corner_2][corner_1] = beam

    def get_axis_beams(self, axis: AxisType) -> List[BeamT]:
        """Returns all non-None beams from given axis"""
        beams = []

        for pair in constants.AXIS_PAIRS[axis]:
            beam = self.beams[pair[0]][pair[1]]

            if beam is not None:
                beams.append(beam)

        return beams

    def get_all_beams(self) -> List[Tuple[int, int, BeamT]]:
        """Returns all non-None entries in self.beams"""
        beams = []
        listed = []

        for corner_1, pairs in enumerate(self.beams):
            for corner_2, beam in pairs.items():
                pair = {corner_1, corner_2}
                if pair in listed:
                    continue

                if beam is not None:
                    beams.append((corner_1, corner_2, beam))
                    listed.append(pair)

        return beams

    def __getitem__(self, index):
        return self.beams[index]
