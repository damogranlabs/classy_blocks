import abc

from classy_blocks.grading.autograding.params import ChopParams, SimpleChopParams
from classy_blocks.mesh import Mesh


class GraderBase(abc.ABC):
    def __init__(self, mesh: Mesh, params: ChopParams):
        self.mesh = mesh
        self.params = params

    @abc.abstractmethod
    def grade(self) -> None:
        pass


class SimpleGrader(GraderBase):
    def __init__(self, mesh: Mesh, params: SimpleChopParams):
        super().__init__(mesh, params)

    def grade(self):
        # just throw the same count into all blocks and be done
        chops = self.params.get_chops_from_length(0)

        for block in self.mesh.blocks:
            for axis in block.axes:
                axis.chops = chops
