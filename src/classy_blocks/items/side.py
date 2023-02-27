"""Contains all data to place a block into mesh."""
import dataclasses

from typing import Optional

from classy_blocks.types import OrientType

@dataclasses.dataclass
class Side:
    """Data about one of block's sides"""
    orient:OrientType

    patch_name:Optional[str] = None # to which patch this block side belongs (if any)
    patch_type:str = 'patch'
    project_to:Optional[str] = None # project to a named searchable surface?
