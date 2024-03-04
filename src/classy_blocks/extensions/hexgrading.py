"""functions specifically used by hexmesh for grading calculations"""

import numpy as np

from classy_blocks.grading.grading import Grading


def grading_ratios(grading: Grading) -> list[float]:
    """Calculates the t_ratios for the grading specification"""

    t_ratios = []

    if grading.is_defined:
        # get the total length ratio
        length_ratio_sum = 0.0
        cell_ratio_sum = 0.0

        for spec in grading.specification:
            length_ratio_sum += spec[0]
            cell_ratio_sum += spec[1]

        if length_ratio_sum == 0.0:
            length_ratio_correction = 1.0
        else:
            length_ratio_correction = 1.0 / length_ratio_sum

        if cell_ratio_sum == 0.0:
            cell_ratio_correction = 1.0
        else:
            cell_ratio_correction = 1.0 / cell_ratio_sum

        length_ratio_sum = 0.0
        for spec in grading.specification:
            # spec[0] = length_ratio
            # spec[1] = cells_ratio
            # spec[2] = expansion

            # FirstToLastCellExpansionRatio
            r_firsttolast = spec[2]  # cmax/cmin

            # Number of cells in this segment
            ncells = int(round(spec[1] * grading.count * cell_ratio_correction, 0))

            # CellToCellExpansionRatio
            r_celltocell = 1.0
            if ncells > 1:
                r_celltocell = np.power(r_firsttolast, 1 / (ncells - 1))

            # width of first cell / segment width
            if r_celltocell == 1.0:
                width_ratio0_seg = 1.0 / ncells
            else:
                width_ratio0_seg = (1.0 - r_celltocell) / (1.0 - r_firsttolast * r_celltocell)

            # load first ratio
            t_ratios.append(length_ratio_sum)

            prev_cell_seg = width_ratio0_seg / r_celltocell
            seg_sum = 0.0
            for _ in range(1, ncells):
                # cell width / segment length
                cell_tr_seg = prev_cell_seg * r_celltocell

                # cell width / length
                cell_tr = cell_tr_seg * spec[0] * length_ratio_correction

                # t_ratio for this cell
                tr = length_ratio_sum + seg_sum + cell_tr
                t_ratios.append(tr)

                prev_cell_seg = cell_tr_seg
                seg_sum += cell_tr

            length_ratio_sum += spec[0] * length_ratio_correction

    else:
        # default to regular spacing

        for it in range(grading.count):
            t_ratios.append(it / grading.count)

    # add last point
    t_ratios.append(grading.count / grading.count)

    return t_ratios
