from typing import Dict, Tuple

from classy_blocks.types import OrientType

# Block WIRE definition:
# according to this sketch: https://www.openfoam.com/documentation/user-guide/blockMesh.php
# the same for all blocks
WIRE_MAP: Dict[OrientType, Tuple[int, int, int, int]] = {
    "bottom": (0, 5, 1, 4),
    "top": (3, 6, 2, 7),
    "left": (4, 11, 7, 8),
    "right": (5, 10, 6, 9),
    "front": (0, 9, 3, 8),
    "back": (1, 10, 2, 11),
}

# defined the two faces for each wire
WIRE_ORIENT: Dict[int, list[OrientType]] = {
    0: ["front", "bottom"],
    1: ["back", "bottom"],
    2: ["back", "top"],
    3: ["front", "top"],
    4: ["left", "bottom"],
    5: ["right", "bottom"],
    6: ["right", "top"],
    7: ["left", "top"],
    8: ["front", "left"],
    9: ["front", "right"],
    10: ["back", "right"],
    11: ["back", "left"],
}

# defined the opposite face for each face
OPPOSITE_FACE: dict[OrientType, OrientType] = {
    "bottom": "top",
    "top": "bottom",
    "left": "right",
    "right": "left",
    "front": "back",
    "back": "front",
}

# define vertex index swap map
RIGHT_TO_LEFT: dict = {0: 1, 3: 2, 7: 6, 4: 5}
FRONT_TO_BACK: dict = {0: 3, 1: 2, 5: 6, 4: 7}
TOP_TO_BOTTOM: dict = {0: 4, 1: 5, 2: 6, 3: 7}
