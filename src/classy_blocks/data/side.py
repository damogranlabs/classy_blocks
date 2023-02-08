"""Contains all data to place a block into mesh."""
import dataclasses

from typing import Optional
from classy_blocks.types import OrientType

@dataclasses.dataclass
class SideData:
    """Data about one of block's sides"""
    orient:OrientType

    patch:Optional[str] = None # whether this block side belongs to a patch
    project:Optional[str] = None # project to a named searchable surface?
