from collections.abc import Sequence
from typing import Union

import numpy as np

from classy_blocks.base.exceptions import InvalidLinkError, NoJunctionError
from classy_blocks.cbtyping import IndexType, NPPointListType, NPPointType
from classy_blocks.construct.assemblies.assembly import Assembly
from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.construct.flat.sketches.mapped import MappedSketch
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.construct.shape import Shape
from classy_blocks.construct.stack import Stack
from classy_blocks.lookup.cell_registry import CellRegistry
from classy_blocks.lookup.connection_registry import (
    ConnectionRegistryBase,
    HexConnectionRegistry,
    QuadConnectionRegistry,
)
from classy_blocks.lookup.face_registry import FaceRegistryBase, HexFaceRegistry, QuadFaceRegistry
from classy_blocks.lookup.point_registry import HexPointRegistry, QuadPointRegistry
from classy_blocks.optimize.cell import CellBase, HexCell, QuadCell
from classy_blocks.optimize.clamps.clamp import ClampBase
from classy_blocks.optimize.junction import Junction
from classy_blocks.optimize.links import LinkBase
from classy_blocks.util import functions as f
from classy_blocks.util.constants import TOL


class GridBase:
    """A list of cells and junctions"""

    cell_class: type[CellBase]
    connection_registry_class: type[ConnectionRegistryBase]
    face_registry_class: type[FaceRegistryBase]

    def __init__(self, points: NPPointListType, addressing: list[IndexType]):
        # work on a fixed point array and only refer to it instead of building
        # new numpy arrays for every calculation
        self.points = points
        self.addressing = addressing

        self.junctions = [Junction(self.points, index) for index in range(len(self.points))]
        self.cells = [self.cell_class(i, self.points, indexes) for i, indexes in enumerate(addressing)]

        self._bind_junction_neighbours()
        self._bind_cell_neighbours()
        self._bind_junction_cells()

    def _bind_junction_neighbours(self) -> None:
        """Adds connections to junctions"""
        creg = self.connection_registry_class(self.points, self.addressing)

        for i, junction in enumerate(self.junctions):
            for c in creg.get_connected_indexes(i):
                junction.neighbours.add(self.junctions[c])

    def _bind_cell_neighbours(self) -> None:
        """Adds neighbours to cells"""
        freg = self.face_registry_class(self.addressing)

        for i, cell in enumerate(self.cells):
            for orient in cell.side_names:
                for neighbour in freg.get_cells(i, orient):
                    if neighbour == i:
                        continue

                    cell.neighbours[orient] = self.cells[neighbour]

    def _bind_junction_cells(self) -> None:
        """Adds cells to junctions"""
        creg = CellRegistry(self.addressing)

        for junction in self.junctions:
            for cell_index in creg.get_near_cells(junction.index):
                junction.cells.add(self.cells[cell_index])

    def get_junction_from_clamp(self, clamp: ClampBase) -> Junction:
        for junction in self.junctions:
            if junction.clamp == clamp:
                return junction

        raise NoJunctionError

    def add_clamp(self, clamp: ClampBase) -> None:
        for junction in self.junctions:
            if f.norm(junction.point - clamp.position) < TOL:
                junction.add_clamp(clamp)
                return

        raise NoJunctionError(f"No junction found for clamp at {clamp.position}")

    def add_link(self, link: LinkBase) -> None:
        leader_index = -1
        follower_index = -1

        for i, junction in enumerate(self.junctions):
            if f.norm(link.leader - junction.point) < TOL:
                leader_index = i
                continue
            if f.norm(link.follower - junction.point) < TOL:
                follower_index = i

        if leader_index == -1:
            raise InvalidLinkError(f"Leader not found for link: {link} (follower: {follower_index})")

        if follower_index == -1:
            raise InvalidLinkError(f"Follower not found for link {link} (leader: {leader_index}")

        if leader_index == follower_index:
            raise InvalidLinkError(f"Leader and follower are the same for link {link} ({leader_index})")

        self.junctions[leader_index].add_link(link, follower_index)

    @property
    def clamps(self) -> list[ClampBase]:
        clamps: list[ClampBase] = []

        for junction in self.junctions:
            if junction.clamp is not None:
                clamps.append(junction.clamp)

        return clamps

    @property
    def quality(self) -> float:
        """Returns summed qualities of all junctions"""
        # It is only called when optimizing linked clamps
        # or at the end of an iteration.
        return sum(junction.quality for junction in self.junctions)

    def update(self, index: int, position: NPPointType) -> float:
        self.points[index] = position

        junction = self.junctions[index]
        quality = junction.quality  # quality is a sum of this junction and all linked ones

        for tie in junction.links:
            # update follower position
            link = tie.leader

            link.leader = position
            link.update()

            # update grid points
            self.points[tie.follower_index] = tie.leader.follower

            # add linked junctions' quality to the sum
            quality += self.junctions[tie.follower_index].quality

        return quality


class QuadGrid(GridBase):
    cell_class = QuadCell
    connection_registry_class = QuadConnectionRegistry
    face_registry_class = QuadFaceRegistry

    @classmethod
    def from_sketch(cls, sketch: Sketch, merge_tol: float = TOL) -> "QuadGrid":
        if isinstance(sketch, MappedSketch):
            # Use the mapper's indexes (provided by the user!)
            return cls(sketch.positions, sketch.indexes)

        # automatically create a mapping for arbitrary sketches
        preg = QuadPointRegistry.from_sketch(sketch, merge_tol)

        return cls(preg.unique_points, preg.cell_addressing)


class HexGrid(GridBase):
    cell_class = HexCell
    connection_registry_class = HexConnectionRegistry
    face_registry_class = HexFaceRegistry

    @classmethod
    def from_elements(
        cls,
        elements: Sequence[Union[Operation, Shape, Stack, Assembly]],
        merge_tol: float = TOL,
    ) -> "HexGrid":
        """Creates a grid from a list of elements"""
        ops: list[Operation] = []
        for element in elements:
            if isinstance(element, Operation):
                ops.append(element)
            else:
                ops += element.operations

        preg = HexPointRegistry.from_operations(ops, merge_tol)

        return cls(preg.unique_points, preg.cell_addressing)

    @classmethod
    def from_mesh(cls, mesh) -> "HexGrid":
        """Creates a grid from an assembled Mesh object"""
        points = np.array([vertex.position for vertex in mesh.vertices])
        addresses = [block.indexes for block in mesh.blocks]

        return cls(points, addresses)
