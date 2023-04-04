import numpy as np
import scipy.optimize

from classy_blocks.util import constants

# here's a list of available calculation functions
# transcribed from the blockmesh grading calculator:
# https://gitlab.com/herpes-free-engineer-hpe/blockmeshgradingweb/-/blob/master/calcBlockMeshGrading.coffee
# (not all are needed in for classy_blocks because length is always a known parameter)
R_MAX = 1 / constants.TOL

# these functions are introspected and used for calculation according to their
# name (get_<result>__<param1>__<param2>(length, param1, param2));
# length is a default argument, passed in always, for simplicity


### functions returning start_size
def get_start_size__count__c2c_expansion(length, count, c2c_expansion):
    """Calculates start size from given count and cell-to-cell expansion ratio"""
    assert length > 0
    assert count >= 1

    if abs(c2c_expansion - 1) > constants.TOL:
        return length * (1 - c2c_expansion) / (1 - c2c_expansion**count)

    return length / count


def get_start_size__end_size__total_expansion(length, end_size, total_expansion):
    """Calculates start size from given end size and total expansion ratio"""
    assert length > 0
    assert total_expansion != 0

    return end_size / total_expansion


### functions returning end_size
def get_end_size__start_size__total_expansion(length, start_size, total_expansion):
    """Calculates end size from given start size and total expansion ratio"""
    assert length > 0

    return start_size * total_expansion


### functions returning count
def get_count__start_size__c2c_expansion(length, start_size, c2c_expansion):
    """Calculates count from given start size and cell-to-cell expansion ratio"""
    assert length > 0
    assert start_size > 0

    if abs(c2c_expansion - 1) > constants.TOL:
        count = np.log(1 - length / start_size * (1 - c2c_expansion)) / np.log(c2c_expansion)
    else:
        count = length / start_size

    return int(count) + 1


def get_count__end_size__c2c_expansion(length, end_size, c2c_expansion):
    """Calculates count from given end size and cell-to-cell expansion ratio"""
    assert length > 0

    if abs(c2c_expansion - 1) > constants.TOL:
        count = np.log(1 / (1 + length / end_size * (1 - c2c_expansion) / c2c_expansion)) / np.log(c2c_expansion)
    else:
        count = length / end_size

    return int(count) + 1


def get_count__total_expansion__c2c_expansion(length, total_expansion, c2c_expansion):
    """Calculates count from total expansion ratio and cell-to-cell expansion ratio"""
    assert length > 0
    assert abs(c2c_expansion - 1) > constants.TOL
    assert total_expansion > 0

    return int(np.log(total_expansion) / np.log(c2c_expansion)) + 1


def get_count__total_expansion__start_size(length, total_expansion, start_size):
    """Calculates count from given total expansion ratio and start size"""
    assert length > 0
    assert start_size > 0
    assert total_expansion > 0

    if total_expansion > 1:
        d_min = start_size
    else:
        d_min = start_size * total_expansion

    if abs(total_expansion - 1) < constants.TOL:
        return int(length / d_min)

    def fcnt(cnt):
        return (1 - total_expansion ** (cnt / (cnt - 1))) / (
            1 - total_expansion ** (1 / (cnt - 1))
        ) - length / start_size

    return int(scipy.optimize.brentq(fcnt, 0, length / d_min)) + 1  # type: ignore


### functions returning c2c_expansion
def get_c2c_expansion__count__start_size(length, count, start_size):
    """Calculates cell-to-cell expansion ratio from given count and start size"""
    assert length > 0
    assert count >= 1
    assert length > start_size > 0

    if count == 1:
        return 1

    if abs(count * start_size - length) / length < constants.TOL:
        return 1

    if count * start_size < length:
        c_max = R_MAX ** (1 / (count - 1))
        c_min = (1 + constants.TOL) ** (1 / (count - 1))
    else:
        c_max = (1 - constants.TOL) ** (1 / (count - 1))
        c_min = (1 / R_MAX) ** (1 / (count - 1))

    def fexp(c2c):
        return (1 - c2c**count) / (1 - c2c) - length / start_size

    if fexp(c_min) * fexp(c_max) >= 0:
        message = "Invalid grading parameters: " + f" length {length}, count {count}, start_size {start_size}"
        raise ValueError(message)

    return scipy.optimize.brentq(fexp, c_min, c_max)


def get_c2c_expansion__count__end_size(length, count, end_size):
    """Calculates cell-to-cell expansion ratio from given count and end size"""
    assert length > 0
    assert count >= 1
    assert end_size > 0

    if abs(count * end_size - length) / length < constants.TOL:
        return 1

    if count * end_size > length:
        c_max = R_MAX ** (1 / (count - 1))
        c_min = (1 + constants.TOL) ** (1 / (count - 1))
    else:
        c_max = (1 - constants.TOL) ** (1 / (count - 1))
        c_min = (1 / R_MAX) ** (1 / (count - 1))

    def fexp(c2c):
        return (1 / c2c ** (count - 1)) * (1 - c2c**count) / (1 - c2c) - length / end_size

    if fexp(c_min) * fexp(c_max) >= 0:
        message = "Invalid grading parameters: " + f" length {length}, count {count}, end_size {end_size}"
        raise ValueError(message)

    return scipy.optimize.brentq(fexp, c_min, c_max)


def get_c2c_expansion__count__total_expansion(length, count, total_expansion):
    """Calculates cell-to-cell expansion ratio from given count and total expansion ratio"""
    assert length > 0
    assert count > 1

    return total_expansion ** (1 / (count - 1))


### functions returning total expansion
def get_total_expansion__count__c2c_expansion(length, count, c2c_expansion):
    """Calculates total expansion ratio from given count and cell-to-cell expansion ratio"""
    assert length > 0
    assert count >= 1

    return c2c_expansion ** (count - 1)


def get_total_expansion__start_size__end_size(length, start_size, end_size):
    """Calculates total expansion ratio from given start size and end size"""
    assert length > 0
    assert start_size > 0
    assert end_size > 0

    return end_size / start_size