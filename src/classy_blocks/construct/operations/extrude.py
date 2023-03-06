import numpy as np

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.loft import Loft

from classy_blocks.types import VectorType

class Extrude(Loft):
    """Takes a Face and extrudes it in given extrude_direction"""
    def __init__(self, base: Face, extrude_vector: VectorType):
        self.base = base
        self.extrude_vector = np.asarray(extrude_vector)

        top_face = base.copy().translate(self.extrude_vector)

        super().__init__(base, top_face)
