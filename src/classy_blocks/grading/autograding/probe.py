import functools
from typing import List, Set, get_args

from classy_blocks.items.block import Block
from classy_blocks.items.wires.axis import Axis
from classy_blocks.mesh import Mesh
from classy_blocks.types import AxisType, ChopTakeType


@functools.lru_cache(maxsize=3000)  # that's for 1000 blocks
def _get_block_from_axis(mesh: Mesh, axis: Axis) -> Block:
    for block in mesh.blocks:
        for index in get_args(AxisType):
            if block.axes[index] == axis:
                return block

    raise RuntimeError("Block for Axis not found!")


class Layer:
    """A collection of all blocks on a specific AxisIndex"""

    def __init__(self, axis: AxisType, blocks: Set[Block]):
        self.axis = axis
        self.blocks = blocks

    def get_length(self, take: ChopTakeType = "avg"):
        lengths: List[float] = []

        for block in self.blocks:
            lengths += block.axes[self.axis].lengths

        if take == "min":
            return min(lengths)

        if take == "max":
            return max(lengths)

        return sum(lengths) / len(self.blocks) / 4  # "avg"


class Catalogue:
    """A collection of layers on a specified axis"""

    def __init__(self, axis: AxisType):
        self.axis = axis
        self.layers: List[Layer] = []

    def has_block(self, block: Block) -> bool:
        for layer in self.layers:
            if block in layer.blocks:
                return True

        return False

    def add_layer(self, blocks: Set[Block]):
        """Adds a block to the appropriate Layer"""
        layer = Layer(self.axis, blocks)
        self.layers.append(layer)

    def get_layer(self, block: Block):
        for layer in self.layers:
            if block in layer.blocks:
                return layer

        # TODO: create a custom exception
        raise RuntimeError("No such layer!")


class Probe:
    """Examines the mesh and gathers required data for auto chopping"""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

        self.catalogues = [Catalogue(axis) for axis in get_args(AxisType)]

        for block in self.mesh.blocks:
            for axis in get_args(AxisType):
                if self.catalogues[axis].has_block(block):
                    continue

                self.catalogues[axis].add_layer(self._get_blocks_on_layer(block, axis))

    def _get_block_from_axis(self, axis: Axis) -> Block:
        return _get_block_from_axis(self.mesh, axis)

    def _get_blocks_on_layer(self, block: Block, axis: AxisType) -> Set[Block]:
        """Returns all blocks on the same 'layer' as the one in arguments"""
        # blocks to be returned
        blocks: Set[Block] = set()
        # blocks not to check again
        traversed: Set[Block] = set()

        def check(blk: Block):
            if blk not in traversed:
                traversed.add(blk)

                for neighbour_axis in blk.axes[axis].neighbours:
                    neighbour_block = self._get_block_from_axis(neighbour_axis)
                    blocks.add(neighbour_block)

                    check(self._get_block_from_axis(neighbour_axis))

        check(block)

        return blocks

    def get_blocks_on_layer(self, block: Block, axis: AxisType):
        return self.catalogues[axis].get_layer(block).blocks

    def get_layers(self, axis: AxisType) -> List[Layer]:
        return self.catalogues[axis].layers
