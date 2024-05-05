# -*- coding: utf-8 -*-
"""
Created on Jun 2023

@author: Henrique Castro
"""

import numpy as np
from base import Element
from sklearn.metrics import mean_squared_error

def sys1_ideal(N):
    w = np.zeros((2+N,1))
    u = np.random.normal(0,1,(N+2,1))
    for i in range(2,N+2):
        w[i] = 0.75*w[i-2] + 0.25*u[i-1] - 0.2*w[i-2]*u[i-1]
    return w[2:],u[2:]

y,u = sys1_ideal(100)
element = Element(weights=(-1,), delays=[1,2,3], nVars=2, nTerms=5, maxHeight=5, mode='MISO')
ind = element.buildModelFromList(['q1(y)','u','mul(q1(y),u)'])
# ind.theta = np.array([0,0.75,0.25,-0.2]).reshape(-1,1)
element.compileModel(ind)
ind.theta = ind.leastSquares(y,u)

"""
Testing predictors and scores
"""
yp,yd = ind.predict("OSA",y,u)
error = ind.score(yd,yp,measure="MSE")
assert  np.isclose(error, 0), "something wrong with miso_OSA/MSE"
error = ind.score(yd,yp,measure="MAPE")
assert  np.isclose(error, 0), "something wrong with miso_OSA/MAPE"

yp,yd = ind.predict("FreeRun",y,u)
error = ind.score(yd,yp[:-1],measure="MSE")
assert  np.isclose(error, 0), "something wrong with miso_FreeRun/MSE"
error = ind.score(yd,yp[:-1],measure="MAPE")
assert  np.isclose(error, 0), "something wrong with miso_FreeRun/MAPE"

yp,yd = ind.predict("MShooting",5,y,u)
error = ind.score(yd,yp,measure="MSE")
assert  np.isclose(error, 0), "something wrong with miso_MShooting/MSE"
error = ind.score(yd,yp,measure="MAPE")
assert  np.isclose(error, 0), "something wrong with miso_MShooting/MAPE"
