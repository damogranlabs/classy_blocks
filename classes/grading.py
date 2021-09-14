import os
import numpy as np

import copy

from ..util import functions as g
from ..util import constants, tools

# In theory, combination of three of these 6 values can be specified:
#  - Total length
#  - Number of cells
#  - Total expansion ratio
#  - Cell-to-cell expansion ratio
#  - Width of start cell
#  - Width of end cell

# In reality, some of them are useless:
#  - Total length: determined by geometry
#  - Total expansion ratio: no physical meaning/usefulness
#  - Width of end cell: same as the above

# Also in reality, some are very important:
#  - Width of start cell: required to keep y+ <= 1
#  - Cell-to-cell expansion ratio: recommended 1.1 - 1.2
#  - Number of cells: the obvious meshing parameter

# Three combinations remain to consider:
#  - Start cell & expansion ratio
#  - Start cell & number of cells
#  - Number of cells & expansion ratio

class Grading:
    """ Grading specification for a single block direction """
    def __init__(self):
        # "multi-grading" specification:
        # https://cfd.direct/openfoam/user-guide/v9-blockMesh/#multi-grading

        self.length_ratios = [] # relative size of block divisions
        self.count_ratios = [] # relative numbers of cells in divisions
        self.expansion_ratios = [] # ratios between start end end cell sizes in divisions
    
    def add_division(self, length_ratio, count_ratio, expansion_ratio):
        self.length_ratios.append(length_ratio)
        self.count_ratios.append(count_ratio)
        self.expansion_ratios.append(expansion_ratio)

    def invert(self):
        # invert gradings and stuff in case neighbuors are defined upside-down
        if len(self.length_ratios) == 0:
            return # nothing to invert
        
        if len(self.length_ratios) == 1:
            self.expansion_ratios[0] = 1/self.expansion_ratios[0]
        
        else:
            raise NotImplementedError("Grading.invert for multigrading")

    def copy(self, invert=False):
        # copy grading from one block to another;
        # use invert=True when neighbours are defined "upside-down"
        g = copy.deepcopy(self)
        if invert:
            g.invert()

        return g

    def __repr__(self):
        if len(self.length_ratios) == 0:
            # no grading specified: default to 1
            return '1'
        
        if len(self.length_ratios) == 1:
            # its a one-number simpleGrading:
            return str(self.expansion_ratios[0])
        
        # 
        s = "(" + os.linesep

        for i in range(len(self.length_ratios)):
            s += f"\t({self.length_ratios[i]} {self.count_ratios[i]} {self.expansion_ratios[i]})"
            s += os.linesep
        
        s += ")"

        return s