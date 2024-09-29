import abc
from typing import get_args

from classy_blocks.grading.autograding.params import ChopParams, SimpleChopParams, SimpleHighReChopParams
from classy_blocks.grading.autograding.probe import Probe
from classy_blocks.mesh import Mesh
from classy_blocks.types import AxisType


class GraderBase(abc.ABC):
    def __init__(self, mesh: Mesh, params: ChopParams):
        self.mesh = mesh
        self.params = params

        self.probe = Probe(self.mesh)

    @abc.abstractmethod
    def grade(self) -> None:
        pass


class FixedCountGrader(GraderBase):
    def __init__(self, mesh: Mesh, params: SimpleChopParams):
        super().__init__(mesh, params)

    def grade(self):
        # just throw the same count into all blocks and be done
        chops = self.params.get_chops_from_length(0)

        for block in self.mesh.blocks:
            for axis in block.axes:
                axis.chops = chops


class SimpleGrader(GraderBase):
    def __init__(self, mesh: Mesh, params: SimpleHighReChopParams):
        super().__init__(mesh, params)

    def grade_axis(self, axis: AxisType) -> None:
        for layer in self.probe.get_layers(axis):
            # TODO: get "take" from the user
            length = layer.get_length("max")
            print(length)
            chops = self.params.get_chops_from_length(length)

            for block in layer.blocks:
                block.axes[axis].chops = chops

    def grade(self):
        for axis in get_args(AxisType):
            self.grade_axis(axis)
