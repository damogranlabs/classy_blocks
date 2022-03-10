import numpy as np
import scipy.optimize

from . import constants

# here's a list of available calculation functions
# transcribed from the blockmesh grading calculator:
# https://gitlab.com/herpes-free-engineer-hpe/blockmeshgradingweb/-/blob/master/calcBlockMeshGrading.coffee
# (not all are needed in for classy_blocks because length is always a known parameter)
r_max = 1/constants.tol

# these functions are introspected and used for calculation according to their
# name (get_<result>__<param1>__<param2>(length, param1, param2));
# length is a default argument, passed in always, for simplicity

### functions returning start_size
def get_start_size__count__c2c_expansion(length, count, c2c_expansion):
    assert length > 0
    assert count >= 1

    h = c2c_expansion - 1
    if abs(h) > constants.tol:
        return length*(1 - c2c_expansion) / (1 - c2c_expansion**count)
    else:
        return length/count

def get_start_size__end_size__total_expansion(length, end_size, total_expansion):
    assert length > 0
    assert total_expansion != 0

    return end_size/total_expansion

### functions returning end_size
def get_end_size__start_size__total_expansion(length, start_size, total_expansion):
    assert length > 0

    return start_size*total_expansion

### functions returning count
def get_count__start_size__c2c_expansion(length, start_size, c2c_expansion):
    assert length > 0
    assert start_size > 0

    if abs(c2c_expansion - 1) > constants.tol:
        count = np.log(
            1 - length/start_size * (1-c2c_expansion)) / \
        np.log(c2c_expansion)
    else:
        count = length/start_size

    return int(count) + 1

def get_count__end_size__c2c_expansion(length, end_size, c2c_expansion):
    assert length > 0

    if abs(c2c_expansion - 1) > constants.tol:
        count = np.log(
            1 / ( 1 + length / end_size * (1 - c2c_expansion)/c2c_expansion)
        )/np.log(c2c_expansion)
    else:
        count = length/end_size

    return int(count) + 1

def get_count__total_expansion__c2c_expansion(length, total_expansion, c2c_expansion):
    assert length > 0
    assert abs(c2c_expansion - 1) > constants.tol
    assert total_expansion > 0

    return int(np.log(total_expansion)/np.log(c2c_expansion)) + 1

def get_count__total_expansion__start_size(length, total_expansion, start_size):
    assert length > 0
    assert start_size > 0
    assert total_expansion > 0

    if total_expansion > 1:
        d_min = start_size
    else:
        d_min = start_size*total_expansion

    if abs(total_expansion - 1) < constants.tol:
        return int(length/d_min)

    fc = lambda n: (1 - total_expansion**(n/(n-1))) / \
        (1 - total_expansion**(1/(n-1))) - length/start_size

    return int(scipy.optimize.brentq(fc, 0, length/d_min)) + 1

### functions returning c2c_expansion
def get_c2c_expansion__count__start_size(length, count, start_size):
    assert length > 0
    assert count >= 1
    assert length > start_size > 0

    if count == 1:
        return 1

    if abs(count*start_size-length)/length < constants.tol:
        return 1
    
    if count*start_size < length:
        c_max = r_max**(1/(count-1))
        c_min = (1 + constants.tol)**(1/(count-1))
    else:
        c_max = (1-constants.tol)**(1/(count-1))
        c_min = (1/r_max)**(1/(count-1))

    fexp = lambda c: (1- c**count) / (1-c) - length/start_size

    if fexp(c_min)*fexp(c_max) >= 0:
        message = "Invalid grading parameters: " + \
        f" length {length}, count {count}, start_size {start_size}"
        raise ValueError(message)

    return scipy.optimize.brentq(fexp, c_min, c_max)

def get_c2c_expansion__count__end_size(length, count, end_size):
    assert length > 0
    assert count >= 1
    assert end_size > 0

    if abs(count*end_size-length)/length < constants.tol:
        return 1
    else:
        if count*end_size > length:
            c_max = r_max**(1/(count-1))
            c_min = (1+constants.tol)**(1/(count-1))
        else:
            c_max= (1-constants.tol)**(1/(count-1))
            c_min= (1/r_max)**(1/(count-1))
        
        fexp = lambda c: (1/c**(count-1))*(1 - c**count)/(1-c)-length/end_size

        if fexp(c_min)*fexp(c_max) >= 0:
            message = "Invalid grading parameters: " + \
                f" length {length}, count {count}, end_size {end_size}"
            raise ValueError(message)

        return scipy.optimize.brentq(fexp, c_min,c_max)

def get_c2c_expansion__count__total_expansion(length, count, total_expansion):
    assert length > 0
    assert count > 1

    return total_expansion**(1/(count-1))
    
### functions returning total expansion
def get_total_expansion__count__c2c_expansion(length, count, c2c_expansion):
    assert length > 0
    assert count >= 1

    return c2c_expansion**(count-1)

def get_total_expansion__start_size__end_size(length, start_size, end_size):
    assert length > 0
    assert start_size > 0
    assert end_size > 0

    return end_size/start_size
