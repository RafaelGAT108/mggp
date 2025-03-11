# -*- coding: utf-8 -*-
"""
Created on Apr 2023

@author: Henrique Castro
"""

from src.base import Element
from src.crossings import CrossLowOnePoint,CrossLowUniform,CrossHighOnePoint,CrossHighUniform
from copy import deepcopy

"""
Test MISO crossings
"""
element = Element(weights=(-1,),delays=[1,2,3],nVars=2,nTerms=5,maxHeight=20,mode='MISO')
element.renameArguments({'ARG0':'y','ARG1':'u'})

ind1 = element.buildRandomModel()
ind2 = element.buildRandomModel()

crossing =  CrossLowOnePoint(element)
ind1_c,ind2_c = crossing.cross(deepcopy(ind1),deepcopy(ind2))
assert True

crossing =  CrossLowUniform(element)
ind1_c,ind2_c = crossing.cross(deepcopy(ind1),deepcopy(ind2))
assert True

crossing =  CrossHighUniform(element)
ind1_c,ind2_c = crossing.cross(deepcopy(ind1),deepcopy(ind2))
assert True

crossing =  CrossHighOnePoint(element)
ind1_c,ind2_c = crossing.cross(deepcopy(ind1),deepcopy(ind2))
assert True

"""
Test MIMO crossings
"""
element = Element(weights=(-1,),delays=[1,2,3],nVars=4,nTerms=5,nOutputs=2,maxHeight=5,mode='MIMO')
element.renameArguments({'ARG0':'y1','ARG1':'y2','ARG2':'u1','ARG3':'u2'})

ind1 = element.buildRandomModel()
ind2 = element.buildRandomModel()

crossing =  CrossLowOnePoint(element)
ind1_c,ind2_c = crossing.cross(deepcopy(ind1),deepcopy(ind2))
assert True

crossing =  CrossLowUniform(element)
ind1_c,ind2_c = crossing.cross(deepcopy(ind1),deepcopy(ind2))
assert True

crossing =  CrossHighOnePoint(element)
ind1_c,ind2_c = crossing.cross(deepcopy(ind1),deepcopy(ind2))
assert True

crossing =  CrossHighUniform(element)
ind1_c,ind2_c = crossing.cross(deepcopy(ind1),deepcopy(ind2))
assert True

