import dataclasses

from classy_blocks.data.chop import Chop

@dataclasses.dataclass
class Division:
    """A collection [length ratio, count ratio, total expansion]
    to be inserted in simpleGrading/edgeGrading part of block specification"""
    chop:Chop

    length_ratio:float
    count:int # just use actual count, blockMesh will normalize them anyway
    total_expansion:float

    # @property
    # def description(self) -> str:
    #     """Output string for blockMeshDict"""
    #     if not self.is_defined:
    #         raise ValueError(f"Grading not defined: {self}")

    #     if len(self.specification) == 1:
    #         # its a one-number simpleGrading:
    #         return str(self.specification[0][2])

    #     # multi-grading: make a nice list
    #     # TODO: make a nicer list
    #     length_ratio_sum = 0
    #     out = "(\n"

    #     for spec in self.specification:
    #         out += f"\t({spec[0]} {spec[1]} {spec[2]})\n"
    #         length_ratio_sum += spec[0]

    #     out += ")"

    #     if length_ratio_sum != 1:
    #         warnings.warn(f"Length ratio doesn't add up to 1: {length_ratio_sum}")

    #     return out