# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 16:56:36 2023

@author: henrique
"""

from src.base import Element
from src.mutations import MutGPOneTree, MutGPUniform, MutGPReplace
from copy import deepcopy

"""
Test MISO mutations
"""
element = Element(weights=(-1,), delays=[1, 2], nVars=2, nTerms=5, maxHeight=3, mode='MISO')
ind = element.buildRandomModel()

mutation = MutGPOneTree(element)
off, = mutation.mutate(deepcopy(ind))
assert True

mutation = MutGPUniform(element)
off, = mutation.mutate(deepcopy(ind))
assert True

mutation = MutGPReplace(element)
off, = mutation.mutate(deepcopy(ind))
assert True

"""
Test MIMO mutations
"""
element = Element(weights=(-1,), delays=[1, 2, 3], nVars=4, nTerms=5, nOutputs=2, maxHeight=5, mode='MIMO')
element.renameArguments({'ARG0': 'y1', 'ARG1': 'y2', 'ARG2': 'u1', 'ARG3': 'u2'})
ind = element.buildRandomModel()

mutation = MutGPOneTree(element)
off, = mutation.mutate(deepcopy(ind))
assert True

mutation = MutGPUniform(element)
off, = mutation.mutate(deepcopy(ind))
assert True

mutation = MutGPReplace(element)
off, = mutation.mutate(deepcopy(ind))
assert True
