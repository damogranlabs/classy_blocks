import dataclasses
import functools
from typing import List, Optional, Set

from classy_blocks.base.exceptions import PatchNotFoundError
from classy_blocks.grading.autograding.catalogue import Catalogue
from classy_blocks.grading.autograding.row import Row
from classy_blocks.items.block import Block
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.wires.wire import Wire
from classy_blocks.mesh import Mesh
from classy_blocks.optimize.grid import HexGrid
from classy_blocks.types import DirectionType, OrientType
from classy_blocks.util.constants import FACE_MAP


@functools.lru_cache(maxsize=2)
def get_defined_wall_vertices(mesh: Mesh) -> Set[Vertex]:
    """Returns vertices that are on the 'wall' patches"""
    wall_vertices: set[Vertex] = set()

    # explicitly defined walls
    for patch in mesh.patches:
        if patch.kind == "wall":
            for side in patch.sides:
                wall_vertices.update(set(side.vertices))

    return wall_vertices


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


class Probe:
    """Examines the mesh and gathers required data for auto chopping"""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

        # maps blocks to rows
        self.catalogue = Catalogue(self.mesh)

        # finds blocks' neighbours
        self.grid = HexGrid.from_mesh(self.mesh)

    def get_row_blocks(self, block: Block, direction: DirectionType) -> List[Block]:
        return self.catalogue.get_row_blocks(block, direction)

    def get_rows(self, direction: DirectionType) -> List[Row]:
        return self.catalogue.rows[direction]

    def get_explicit_wall_vertices(self, block: Block) -> Set[Vertex]:
        """Returns vertices from a block that lie on explicitly defined wall patches"""
        mesh_vertices = get_defined_wall_vertices(self.mesh)
        block_vertices = set(block.vertices)

        return block_vertices.intersection(mesh_vertices)

    def get_default_wall_vertices(self, block: Block) -> Set[Vertex]:
        """Returns vertices that lie on default 'wall' patch"""
        wall_vertices: Set[Vertex] = set()

        if "kind" not in self.mesh.patch_list.default:
            # the mesh has no default wall patch
            return wall_vertices

        # other sides when mesh has a default wall patch
        if self.mesh.patch_list.default["kind"] == "wall":
            # find block boundaries
            block_index = self.mesh.blocks.index(block)
            cell = self.grid.cells[block_index]

            # sides with no neighbours are on boundary
            boundaries: List[OrientType] = [
                orient for orient, neighbours in cell.neighbours.items() if neighbours is None
            ]

            for orient in boundaries:
                side_vertices = {block.vertices[i] for i in FACE_MAP[orient]}
                # check if they are defined elsewhere
                try:
                    self.mesh.patch_list.find(side_vertices)
                    # the patch is defined elsewhere and is not included here among default ones
                    continue
                except PatchNotFoundError:
                    wall_vertices.update(side_vertices)

        return wall_vertices

    def get_wall_vertices(self, block: Block) -> Set[Vertex]:
        """Returns vertices that are on the 'wall' patches"""
        return self.get_explicit_wall_vertices(block).union(self.get_default_wall_vertices(block))

    def get_wire_info(self, wire: Wire, block: Block) -> WireInfo:
        # TODO: test
        wall_vertices = self.get_wall_vertices(block)

        return WireInfo(wire, wire.vertices[0] in wall_vertices, wire.vertices[1] in wall_vertices)
