# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 12:31:54 2023

@author: henrique
"""

import sys
from numba import njit
sys.path.insert(1, 'MGGP/')
import time
import numpy as np
import matplotlib.pyplot as plt
from base import Element
from evolvers import EvolDefault
import multiprocessing
from sklearn.metrics import mean_squared_error

'''
## SISO models optimization

This file presents an example of the MGGP toolbox for identification of SISO models.

Consider the following model to be identified:
'''
def calculate_result(k):
        if k >= 1:
            model = 1 + y[k - 1] + u[k] * y[k - 1] + u[k]
            print(f"y in k={k} is {y[k]} and de model is {model}. "
                  f"The MSE between y and model is {mean_squared_error(y[k], model)} ")
        else:
            print("K must be >= 1")

def sys(N):
    y1 = np.zeros((3 + N, 1))
    u1 = np.random.normal(0, 1, (N + 3, 1))
    for i in range(3, N + 3):
        y1[i] = 0.75 * y1[i - 2] + 0.25 * u1[i - 1] - 0.2 * y1[i - 2] * u1[i - 1]
    return y1[3:], u1[3:]


y, u = sys(1003)

'''
The first step is to configure the Element object. It is responsible to 
carry all information about individual (model) features, 
and to create and compile it.

The Element class receives seven arguments:
(with default setting)<br />
1. wheights  = (-1,) - defines the type of optimization (negative for minimization, 
                        positive for maximization). It must be a tupple.
2. delays    = [1,2,3] - a list of positive integer numbers that will define the 
                        backshift operators q^{-n}, for n in delays.
3. nInputs   = 2 - number of inputs in the system.
4. nOutputs  = 1 - number of outputs in the system.
5. nTerms    = 10 - number of terms each output model will possess.
6. maxHeight = 5 - maximum height of GP individual.
7. mode      = "MISO" - a string specifying the type of model the MGGP algorithm must identify.

The mode argument shall be "MISO" or "MIMO"

For this example, let the Element object be instanciated as
'''
element = Element(weights=(-1,), delays=[1, 2, 3], nInputs=1, nOutputs=1,
                  nTerms=3, maxHeight=5, mode="MISO")
element.renameArguments({'ARG0': 'y', 'ARG1': 'u'})


'''
Now, define a cost function to assess individuals. The evaluation function receives 
as arguments only the individual object.
To aid in this task, the individual objects have the built-in methods 
predict(mode,*args) and score(yd,yp,mode) methods:
    
* predict(mode,*args)
  + mode = ["OSA","FreeRun","MShooting"] - one-step-ahear, free-run and multiple-shooting predictors
  + yp,yd = predict("OSA",y,u)
  + yp,yd = predict("FreeRun",y,u)
  + yp,yd = predict("MShooting",k,y,u)<br />
in which yp stands for predicted and yd for desired. The argument k in the 
"MShooting" mode determines the number of prediction steps.

* score(yd,yp,mode)
  + mode = ["MSE","MAPE"]
in which "MSE" stands for mean squared error and "MAPE" for mean absolute percentile error.

Before prediction, the individuals must be compiled. This task belongs to the Element object.
'''


def evaluation(ind):
    try:
        element.compileModel(ind)
        ind.theta = ind.leastSquares(y, u)
        yp, yd = ind.predict("OSA", y, u)
        error = ind.score(yd, yp, "MSE")
        return error,
    except np.linalg.LinAlgError:
        return (np.inf,)
    # except ValueError:
    #     return (np.inf,)


# an exception treatment must be placed to avoid interruptions due to singular or ill conditioned regressors matrices

'''
Next, define the Evolver object. It is implemented only the EvolDeault class 
that receives as arguments

1. element - Element object previously defined.
2. evaluate - cost function previously defined.
3. popSize - population size.
4. elitePerc - percentile of population to be kept in the next generation.
5. CXPB - crossover probability.
6. MTPB - mutation probability.

For this example, let the Evolver object be
'''

evolver = EvolDefault(element=element, evaluate=evaluation,
                      popSize=100, elitePerc=10, CXPB=0.8, MTPB=0.1)

'''
Finally, the algorithm can be run. The Evolver object do the task.

Follow the example:
* use multiprocessing (process-based parallelism).
* evolver.initPop() - initiate population.
* evolver.stream() = print current state of evolution logbook.
* evolver.step() = go to the next generation.
* evolver.getHof() = get elite population.
* hof[0] = best individual.
'''

if __name__ == "__main__":

    #pool = multiprocessing.Pool(4)  # using 4 processor cores
    #evolver._toolbox.register("map", pool.map)
    evolver._toolbox.register("map", map)
    init = time.time()
    evolver.initPop()
    evolver.stream()

    for g in range(100):
        evolver.step()
        evolver.stream()

    end = time.time()
    hof = evolver.getHof()
    model = hof[0]

    # element.compileModel(model)
    # model.theta = model.leastSquares(y, u)

    print(model)

    print(f"time: {round(end - init, 3)} seg")
    # yp,yd = model.predict("FreeRun",y,u)

    # plt.figure()
    # plt.plot(yd[:,0])
    # plt.plot(yd[:,0])

    # plt.figure()
    # plt.plot(yd[:,1])
    # plt.plot(yd[:,1])

    # def calculate_result(k):
    #     if k >= 1:
    #         model = 1 + y[k - 1] + u[k] * y[k - 1] + u[k]
    #         print(f"y in k={k} is {y[k]} and de model is {model}")
    #     else:
    #         print("K must be >= 1")
