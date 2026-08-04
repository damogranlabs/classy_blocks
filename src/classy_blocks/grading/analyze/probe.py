import dataclasses
from typing import Optional

from classy_blocks.assemble.dump import AssembledDump
from classy_blocks.assemble.settings import Settings
from classy_blocks.base.exceptions import PatchNotFoundError
from classy_blocks.cbtyping import DirectionType, OrientType
from classy_blocks.grading.analyze.catalogue import RowCatalogue
from classy_blocks.grading.analyze.row import Row
from classy_blocks.items.block import Block
from classy_blocks.items.wires.wire import Wire
from classy_blocks.optimize.grid import HexGrid
from classy_blocks.util.constants import DIRECTION_MAP


@dataclasses.dataclass
class WireInfo:
    """Gathers data about a wire; its location and whether it starts or ends on a wall"""

    wire: Wire
    starts_at_wall: bool
    ends_at_wall: bool


class WireCatalogue:
    """A database of all wires' whereabouts;
    many wires can be located at the same spot and only some of them can be
    on 'walls'; gather data first so that wall-bounded wires aren't ignored"""

    def __init__(self, dump: AssembledDump, settings: Settings):
        self.dump = dump
        self.settings = settings

        # finds blocks' neighbours
        self.grid = HexGrid.from_dump(dump)

        # WireInfo is stored at indexes [vertex_1][vertex_2];
        # if a coincident wire is inverted, it will reside at
        # [vertex_2][vertex_1]
        self.data: dict[int, dict[int, Optional[WireInfo]]] = {}

        self._populate()

    def _fetch(self, index_1: int, index_2: int) -> Optional[WireInfo]:
        if self.data.get(index_1) is None:
            self.data[index_1] = {}

        return self.data[index_1].get(index_2)

    def _populate(self) -> None:
        for block in self.dump.blocks:
            for wire in block.wire_list:
                start_index, end_index = wire.vertices[0].index, wire.vertices[1].index

                info = self._fetch(start_index, end_index)
                if info is None:
                    info = WireInfo(wire, False, False)
                    self.data[start_index][end_index] = info

                starts, ends = self._get_wire_boundaries(wire, block)

                # remember if any of the coincident wires starts or ends at the wall,
                info.starts_at_wall = info.starts_at_wall or starts
                info.ends_at_wall = info.ends_at_wall or ends

    def _get_wire_boundaries(self, wire: Wire, block: Block) -> tuple[bool, bool]:
        """Finds out whether a Wire starts or ends on a wall patch"""
        # only check  patches, perpendicular to the wire;
        # wires lying on wall surfaces aren't eligible for wall/inflation grading
        start_orient = DIRECTION_MAP[wire.direction][0]
        end_orient = DIRECTION_MAP[wire.direction][1]
        block_index = self.dump.blocks.index(block)

        def is_on_wall(orient: OrientType) -> bool:
            if self.grid.cells[block_index].neighbours[orient] is not None:
                # this block side has a neighbour and cannot be a wall patch
                return False

            vertices = set(block.get_side_vertices(orient))

            try:
                patch = self.dump.patch_list.find(vertices)
                if patch.kind == "wall":
                    return True
            except PatchNotFoundError:
                if self.settings.default_patch.get("kind") == "wall":
                    return True

            return False

        return (is_on_wall(start_orient), is_on_wall(end_orient))

    def get_info(self, wire: Wire) -> WireInfo:
        info = self.data[wire.vertices[0].index][wire.vertices[1].index]

        if info is None:
            raise ValueError("Wire not found!")

        return info


class Probe:
    """Examines the mesh and gathers required data for auto chopping"""

    def __init__(self, dump: AssembledDump, settings: Settings):
        # maps blocks to rows
        self.rows = RowCatalogue(dump.block_list)

        # build a wire database
        self.wires = WireCatalogue(dump, settings)

    def get_row_blocks(self, block: Block, direction: DirectionType) -> list[Block]:
        return self.rows.get_row_blocks(block, direction)

    def get_rows(self, direction: DirectionType) -> list[Row]:
        return self.rows.rows[direction]

    def get_wire_info(self, wire: Wire) -> WireInfo:
        return self.wires.get_info(wire)
