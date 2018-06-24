'''Utility functions for network moments.'''
from .utils import (epsilon, diagonal, mul_diag, outer,
                    normalize, normalize_, flatten,
                    jacobian, linearize,
                    rand_from_eigen, rand_definite)

import sys
from types import ModuleType
rand = ModuleType('rand', 'Random tensors generators.')
rand.definite = rand_definite
rand.from_eigen = rand_from_eigen
sys.modules[rand.__name__] = rand

del (utils, rand_definite, rand_from_eigen, sys, ModuleType)
