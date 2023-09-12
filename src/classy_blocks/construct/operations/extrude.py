from typing import Union

import numpy as np

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.types import VectorType


class Extrude(Loft):
    """Takes a Face and extrudes it by 'amount'.
    If 'amount' is float, the extrude direction is normal to 'base'.
    """

    def __init__(self, base: Face, amount: Union[float, VectorType]):
        self.base = base

        if isinstance(amount, float) or isinstance(amount, int):
            extrude_vector = self.base.normal * amount
        else:
            extrude_vector = np.asarray(amount)

        top_face = base.copy().translate(extrude_vector)

        super().__init__(base, top_face)
