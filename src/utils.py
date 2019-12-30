from warnings import warn
from collections.abc import MutableSequence
import numpy as np

def isiterable(obj):
    if isinstance(obj, str): return False
    try:
        iter(obj)
        return True
    except TypeError: return False

def isiterable_till(obj, depth=1):
    assert(depth>=0)
    if depth==0: return not isiterable(obj)
    else: return isiterable(obj) and all([isiterable_till(o, depth=depth-1) for o in obj])

def istype(obj, type):
    if isiterable(obj): return all([istype(o, type) for o in obj])
    if isinstance(obj, type): return True
    else: return False

def isnumeric(obj): return istype(obj, (int, float, np.integer, np.float))

def isstring(obj): return istype(obj, str)
