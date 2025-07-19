import dataclasses
from typing import Optional

from classy_blocks.assemble.dump import AssembledDump
from classy_blocks.base.exceptions import PatchNotFoundError
from classy_blocks.cbtyping import DirectionType, OrientType
from classy_blocks.grading.autograding.catalogue import RowCatalogue
from classy_blocks.grading.autograding.row import Row
from classy_blocks.items.block import Block
from classy_blocks.items.wires.wire import Wire
from classy_blocks.mesh import Mesh
from classy_blocks.optimize.grid import HexGrid
from classy_blocks.util.constants import DIRECTION_MAP


@dataclasses.dataclass
class WireInfo:
    """Gathers data about a wire; its location, cell sizes, neighbours and wires before/after"""

    wire: Wire
    starts_at_wall: bool
    ends_at_wall: bool

    @property
    def length(self) -> float:
        return self.wire.length

    @property
    def size_after(self) -> Optional[float]:
        """Returns average cell size in wires that come after this one (in series/inline);
        None if this is the last wire"""
        # TODO: merge this with size_before somehow
        sum_size: float = 0
        defined_count: int = 0

        for joint in self.wire.after:
            if joint.wire.grading.is_defined:
                defined_count += 1

                if joint.same_dir:
                    sum_size += joint.wire.grading.start_size
                else:
                    sum_size += joint.wire.grading.end_size

        if defined_count == 0:
            return None

        return sum_size / defined_count

    @property
    def size_before(self) -> Optional[float]:
        """Returns average cell size in wires that come before this one (in series/inline);
        None if this is the first wire"""
        # TODO: merge this with size_after somehow
        sum_size: float = 0
        defined_count: int = 0

        for joint in self.wire.before:
            if joint.wire.grading.is_defined:
                defined_count += 1

                if joint.same_dir:
                    sum_size += joint.wire.grading.end_size
                else:
                    sum_size += joint.wire.grading.start_size

        if defined_count == 0:
            return None

        return sum_size / defined_count


class WireCatalogue:
    """A database of all wires' whereabouts;
    many wires can be located at the same spot and only some of them can be
    on 'walls'; gather data first so that wall-bounded wires aren't ignored"""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        assert isinstance(self.mesh.dump, AssembledDump)
        self.dump = self.mesh.dump

        # finds blocks' neighbours
        self.grid = HexGrid.from_mesh(self.mesh)

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
        for block in self.mesh.blocks:
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
        start_orient = DIRECTION_MAP[wire.direction][0]
        end_orient = DIRECTION_MAP[wire.direction][1]
        block_index = self.mesh.blocks.index(block)

        def find_patch(orient: OrientType) -> bool:
            # search for external faces;
            # either this block has one or any of its neighbours at the start
            # or end of this wire
            if self.grid.cells[block_index].neighbours[orient] is not None:
                # Internal face
                return False

            vertices = set(block.get_side_vertices(orient))
            try:
                patch = self.dump.patch_list.find(vertices)
                if patch.kind == "wall":
                    return True
            except PatchNotFoundError:
                if self.mesh.settings.default_patch.get("kind") == "wall":
                    return True

            return False

        return (find_patch(start_orient), find_patch(end_orient))

    def get_info(self, wire: Wire) -> WireInfo:
        info = self.data[wire.vertices[0].index][wire.vertices[1].index]

        if info is None:
            raise ValueError("Wire not found!")

        return info


class Probe:
    """Examines the mesh and gathers required data for auto chopping"""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

        # maps blocks to rows
        self.rows = RowCatalogue(self.mesh)

        # build a wire database
        self.wires = WireCatalogue(self.mesh)

    def get_row_blocks(self, block: Block, direction: DirectionType) -> list[Block]:
        return self.rows.get_row_blocks(block, direction)

    def get_rows(self, direction: DirectionType) -> list[Row]:
        return self.rows.rows[direction]

    def get_wire_info(self, wire: Wire) -> WireInfo:
        return self.wires.get_info(wire)
