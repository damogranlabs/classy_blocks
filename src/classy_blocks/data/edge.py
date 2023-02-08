"""Contains all data that user can specify for any kind of Edge"""
import dataclasses

from classy_blocks.types import EdgeKindType

@dataclasses.dataclass
class EdgeData:
    """User-provided data for an edge
    Constructor args:
    :param index_1: block-local index to block.points
    :param index_2: block-local index to block.points
    :param kind: edge type
    :param data: optional additional arguments for given edge"""
    index_1:int
    index_2:int
    kind:EdgeKindType
    args:list
