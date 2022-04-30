import os

import copy

from ..util import grading_calculator as gc
import inspect
import warnings

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

# Also in reality, some are very important:
#  - Width of start cell: required to keep y+ <= 1
#  - Cell-to-cell expansion ratio: 1 < recommended <= 1.2
#  - Number of cells: the obvious meshing parameter
#  - Width of end cell: matching with cell sizes of the next block

# Possible Combinations to consider are:
#  - start_size & c2c_expansion
#  - start_size & count
#  - start_size & end_size
#  - end_size & c2c_expansion
#  - end_size & count
#  - count & c2c_expansion

# Some nomenclature to avoid confusion
# Grading: the whole grading specification (length, count, expansion ratio)
# length: length of an edge we're dealing with
# count: number of cells in a block for a given direction
# total_expansion: ratio between last/first cell size
# c2c_expansion: ratio between sizes of two neighbouring cells
# start_size: size of the first cell in a block
# end_size: size of the last cell in a block
# division: an entry in simpleGrading specification in blockMeshDict

# calculations meticulously transcribed from the blockmesh grading calculator:
# https://gitlab.com/herpes-free-engineer-hpe/blockmeshgradingweb/-/blob/master/calcBlockMeshGrading.coffee
# (since block length is always known, there's less wrestling but the calculation principle is similar)

# gather available functions for calculation of grading parameters
functions = [] # list [ [return_value, {parameters}, function], ... ]

for m in inspect.getmembers(gc):
    if inspect.isfunction(m[1]):
        # function name is assembled as
        # get_<result>__<param1>__<param2>
        data = m[0].split(sep='__')
        functions.append(
            [data[0][4:], [data[1], data[2]], m[1]]
        )

def calculate(length, parameters):
    # calculates cell count and total expansion ratio for a block
    # by calling functions that take known variables and return new values
    keys = parameters.keys()
    calculated = set()

    for k in keys:
        if parameters[k] is not None:
            calculated.add(k)

    for _ in range(20):
        for f in functions:
            freturn = f[0]
            fparams = f[1]
            ffunction = f[2]
            if freturn in calculated:
                # this value is already calculated, go on
                continue
    
            if set(fparams).issubset(calculated):
                parameters[freturn] = ffunction(length, parameters[fparams[0]], parameters[fparams[1]])
                calculated.add(freturn)

        if {'count', 'total_expansion'}.issubset(calculated):
            return parameters['count'], parameters['total_expansion']
    
    raise ValueError(f"Could not calculate count and grading for given parameters: {parameters}")

class Grading:
    """ Grading specification for a single block direction """
    def __init__(self):
        # must be set before any calculation is performed
        self.length = None

        # "multi-grading" specification according to:
        # https://cfd.direct/openfoam/user-guide/v9-blockMesh/#multi-grading
        self.divisions = [] # a list of lists [length ratio, count ratio, total expansion]
    
    def set_block_size(self, size):
        self.length = size

    def add_division(self, length_ratio=1,
        start_size=None, c2c_expansion=None, count=None, end_size=None, total_expansion=None, invert=False):
        """ Add a grading division to block specification.
        Use length_ratio for multigrading (see documentation).
        Available grading parameters are:
         - start_size: width of start cell
         - end_size: width of end cell
         - count: cell count in given direction
         - c2c_expansion: cell-to-cell expansion ratio (default=1)
         - total_expansion: ratio between first and last cell size

        You must specify start_size and/or count.
        c2c_expansion is optional - will be used to create graded cells
        and will default to 1 if not provided.

        To reverse grading, use invert=True.

        Documentation:
        https://cfd.direct/openfoam/user-guide/v9-blockMesh/#multi-grading """
        assert self.length is not None
        assert 0 < length_ratio <= 1

        # default: take c2c_expansion=1 if there's less than 2 parameters given
        grading_params = [start_size, end_size, c2c_expansion, count]
        if grading_params.count(None) > len(grading_params) - 2:
            c2c_expansion = 1

        # also: count can only be an integer
        if count is not None:
            count = int(count)

        length = self.length*length_ratio

        # blockMesh needs two numbers regardless of user's input:
        # number of cells and total expansion ratio.
        # those two can be calculated from count and c2c_expansion
        # so calculate those first
        count, total_expansion = calculate(length, {
            'start_size': start_size,
            'end_size': end_size,
            'c2c_expansion': c2c_expansion,
            'count': count,
            'total_expansion': total_expansion
        })

        if invert:
            total_expansion = 1/total_expansion

        self.divisions.append([length_ratio, count, total_expansion])

    def invert(self):
        # invert gradings and stuff in case neighbuors are defined upside-down
        if len(self.divisions) == 0:
            return # nothing to invert 
        
        if len(self.divisions) == 1:
            self.divisions[0][2] = 1/self.divisions[0][2]
        else:
            # divisions: a list of lists [length ratio, count ratio, total expansion]
            d_inverted = []

            for d in reversed(self.divisions):
                d[2] = 1/d[2]
                d_inverted.append(d)
                self.divisions = d_inverted

    def copy(self, invert=False):
        # copy grading from one block to another;
        # use invert=True when neighbours are defined "upside-down"
        g = copy.deepcopy(self)
        if invert:
            g.invert()

        return g

    @property
    def count(self):
        return sum([d[1] for d in self.divisions])

    @property
    def is_defined(self):
        # grading is defined if there's at least one division added
        return len(self.divisions) > 0

    def __repr__(self):
        if len(self.divisions) == 0:
            # no grading specified: default to 1
            return 'Undefined'
        
        if len(self.divisions) == 1:
            # its a one-number simpleGrading:
            return str(self.divisions[0][2])
        
        length_ratio_sum = 0
        s = "(" + os.linesep

        for d in self.divisions:
            s += f"\t({d[0]} {d[1]} {d[2]})"
            s += os.linesep

            length_ratio_sum += d[0]
        
        s += ")"

        if length_ratio_sum != 1:
            warnings.warn(f"Length ratio doesn't add up to 1: {length_ratio_sum}")

        return s
    