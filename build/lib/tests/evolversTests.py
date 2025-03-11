# -*- coding: utf-8 -*-
"""
Created on Apr 2023

@author: Henrique Castro
"""

from src.base import Element
from src.evolvers import EvolDefault
import numpy as np
import multiprocessing
import time

def evaluate(ind):
    element.compileModel(ind)
    return np.random.random(),

# element = Element(weights=(-1,),delays=[1,2,3],nVars=2,nTerms=5,maxHeight=20,mode='MISO')
# element.renameArguments({'ARG0':'y','ARG1':'u'})

# element = Element(weights=(-1,), delays=[1, 2, 3], nVars=4, nTerms=5, nOutputs=2, maxHeight=5, mode='MIMO')

element = Element(weights=(-1,), delays=[1, 2, 3],
                  nInputs=2, nTerms=5, nOutputs=2,
                  maxHeight=5, mode='MIMO')
element.renameArguments({'ARG0': 'y1', 'ARG1': 'y2', 'ARG2': 'u1', 'ARG3': 'u2'})

evolver = EvolDefault(element=element, evaluate=evaluate,
                      popSize=100, elitePerc=10, CXPB=0.8, MTPB=0.1)

if __name__ == "__main__":

    pool = multiprocessing.Pool(7)
    evolver._toolbox.register("map", pool.map)
    #evolver._toolbox.register("map", map)

    init = time.time()
    evolver.initPop()
    evolver.stream()

    for g in range(100):
        evolver.step()
        evolver.stream()
    end = time.time()

    print(f"time: {round(end - init, 3)} seg")
