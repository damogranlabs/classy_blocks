from classy_blocks.grading.autograding.graders import GraderBase
from classy_blocks.grading.autograding.inflation.rules import InflationRules
from classy_blocks.mesh import Mesh


class InflationGrader(GraderBase):
    """Parameters for mesh grading for Low-Re cases.
    To save on cell count, only a required thickness (inflation layer)
    will be covered with thin cells (c2c_expansion in size ratio between them).
    Then a bigger expansion ratio will be applied between the last cell of inflation layer
    and the first cell of the bulk flow.

    Example:
     ________________
    |
    |                 > bulk size (cell_size=bulk, no expansion)
    |________________
    |
    |________________ > buffer layer (c2c = 2)
    |________________
    |================ > inflation layer (cell_size=y+, c2c=1.2)
    / / / / / / / / / wall

    Args:
        first_cell_size (float): thickness of the first cell near the wall
        c2c_expansion (float): expansion ratio between cells in inflation layer
        bl_thickness_factor (int): thickness of the inflation layer in y+ units (relative to first_cell_size)
        buffer_expansion (float): expansion between cells in buffer layer
        bulk_cell_size (float): size of cells inside the domain

        Autochop will take all relevant blocks and choose one to start with - set cell counts
        and other parameters that must stay fixed for all further blocks.
        It will choose the longest/shortest ('max/min') block edge or something in between ('avg').
        The finest grid will be obtained with 'max', the coarsest with 'min'.
    """

    def __init__(
        self,
        mesh: Mesh,
        first_cell_size: float,
        bulk_cell_size: float,
        c2c_expansion: float = 1.2,
        bl_thickness_factor: int = 30,
        buffer_expansion: float = 2,
    ):
        params = InflationRules(first_cell_size, bulk_cell_size, c2c_expansion, bl_thickness_factor, buffer_expansion)

        super().__init__(mesh, params)
