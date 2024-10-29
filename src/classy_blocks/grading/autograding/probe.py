import functools
from typing import Dict, List, Optional, Set, get_args

from classy_blocks.base.exceptions import BlockNotFoundError, NoInstructionError
from classy_blocks.items.block import Block
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.wires.axis import Axis
from classy_blocks.items.wires.wire import Wire
from classy_blocks.mesh import Mesh
from classy_blocks.optimize.grid import HexGrid
from classy_blocks.types import ChopTakeType, DirectionType, OrientType
from classy_blocks.util.constants import FACE_MAP


@functools.lru_cache(maxsize=3000)  # that's for 1000 blocks
def get_block_from_axis(mesh: Mesh, axis: Axis) -> Block:
    for block in mesh.blocks:
        if axis in block.axes:
            return block

    raise RuntimeError("Block for Axis not found!")


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


class Instruction:
    """A descriptor that tells in which direction the specific block can be chopped."""

    def __init__(self, block: Block):
        self.block = block
        self.directions: List[bool] = [False] * 3

    @property
    def is_defined(self):
        return all(self.directions)

    def __hash__(self) -> int:
        return id(self)


class Row:
    """A string of blocks that must share the same count
    because they sit next to each other.

    This needs not be an actual 'string' of blocks because,
    depending on blocking, a whole 'layer' of blocks can be
    chopped by specifying a single block only (for example, direction 2 in a cylinder)"""

    def __init__(self, direction: DirectionType):
        self.direction = direction

        # blocks of different orientations can belong to the same row;
        # remember how they are oriented
        self.blocks: List[Block] = []
        self.headings: List[DirectionType] = []

    def add_block(self, block: Block, row_direction: DirectionType) -> None:
        self.blocks.append(block)
        self.headings.append(row_direction)

    def get_length(self, take: ChopTakeType = "avg"):
        lengths = [wire.length for wire in self.get_wires()]

        if take == "min":
            return min(lengths)

        if take == "max":
            return max(lengths)

        return sum(lengths) / len(self.blocks) / 4  # "avg"

    def get_axes(self) -> List[Axis]:
        axes: List[Axis] = []

        for i, block in enumerate(self.blocks):
            direction = self.headings[i]
            axes.append(block.axes[direction])

        return axes

    def get_wires(self) -> List[Wire]:
        wires: List[Wire] = []

        for axis in self.get_axes():
            wires += axis.wires

        return wires

    def get_count(self) -> Optional[int]:
        for wire in self.get_wires():
            if wire.is_defined:
                return wire.grading.count

        return None


class Catalogue:
    """A collection of rows on a specified axis"""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

        self.rows: Dict[DirectionType, List[Row]] = {0: [], 1: [], 2: []}
        self.instructions = [Instruction(block) for block in mesh.blocks]

        for i in get_args(DirectionType):
            self._populate(i)

    def _get_undefined_instructions(self, direction: DirectionType) -> List[Instruction]:
        return [i for i in self.instructions if not i.directions[direction]]

    def _find_instruction(self, block: Block):
        # TODO: perform dedumbing on this exquisite piece of code
        for instruction in self.instructions:
            if instruction.block == block:
                return instruction

        raise NoInstructionError(f"No instruction found for block {block}")

    def _add_block_to_row(self, row: Row, instruction: Instruction, direction: DirectionType) -> None:
        row.add_block(instruction.block, direction)
        instruction.directions[direction] = True

        block = instruction.block

        for neighbour_axis in block.axes[direction].neighbours:
            neighbour_block = get_block_from_axis(self.mesh, neighbour_axis)

            if neighbour_block in row.blocks:
                continue

            instruction = self._find_instruction(neighbour_block)

            self._add_block_to_row(row, instruction, neighbour_block.get_axis_direction(neighbour_axis))

    def _populate(self, direction: DirectionType) -> None:
        while True:
            undefined_instructions = self._get_undefined_instructions(direction)
            if len(undefined_instructions) == 0:
                break

            row = Row(direction)
            self._add_block_to_row(row, undefined_instructions[0], direction)
            self.rows[direction].append(row)

    def get_row_blocks(self, block: Block, direction: DirectionType) -> List[Block]:
        for row in self.rows[direction]:
            if block in row.blocks:
                return row.blocks

        raise BlockNotFoundError(f"Direction {direction} of {block} not in catalogue")


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
                wall_vertices.union({block.vertices[i] for i in FACE_MAP[orient]})

        return wall_vertices

    def get_wall_vertices(self, block: Block) -> Set[Vertex]:
        """Returns vertices that are on the 'wall' patches"""
        return self.get_explicit_wall_vertices(block).union(self.get_default_wall_vertices(block))
