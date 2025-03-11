# -*- coding: utf-8 -*-
"""
Created on Jun 2023

@author: Henrique Castro
"""

import numpy as np
from src.base import Element


def sys(N):
    w1 = np.zeros((3 + N, 1))
    u1 = np.random.normal(0, 1, (N + 3, 1))

    w2 = np.zeros((3 + N, 1))
    u2 = np.random.normal(0, 1, (N + 3, 1))

    for i in range(3, N + 3):
        w1[i] = 0.75 * w1[i - 2] + 0.25 * u1[i - 1] - 0.2 * w1[i - 2] * u2[i - 1] + 0.1 * w1[i - 3] * w2[i - 2]

        w2[i] = 0.4 * u2[i - 1] ** 2 + 0.6 * w2[i - 1] - 0.45 * w2[i - 2] - 0.1 * w1[i - 1] - 0.2

    y = np.concatenate([w1, w2], axis=1)
    u = np.concatenate([u1, u2], axis=1)
    return y[3:], u[3:]


element = Element(weights=(-1,), delays=[1, 2, 3], nInputs=2, nOutputs=2, nTerms=5, maxHeight=5, mode='MIMO')
element.renameArguments({'ARG0': 'y1', 'ARG1': 'y2', 'ARG2': 'u1', 'ARG3': 'u2'})

ind = element.buildModelFromList([['q1(y1)', 'u1', 'mul(q1(y1),u2)', 'mul(q2(y1),q1(y2))'],
                                  ['mul(u2,u2)', 'y2', 'q1(y2)', 'y1']])

# ind.theta = np.array([[0,0.75,0.25,-0.2,0.1],
#                       [-0.2,0.4,.6,-0.45,-0.1]]).T

element.compileModel(ind)

"""
Testing predictors and scores
"""

y, u = sys(205)

ind.theta = ind.leastSquares(y, u)

yp, yd = ind.predict("OSA", y, u)
error = ind.score(yd, yp, measure="MSE")
assert np.isclose(error, 0), "something wrong with mimo_OSA/MSE"
error = ind.score(yd, yp, measure="MAPE")
assert np.isclose(error, 0), "something wrong with mimo_OSA/MAPE"

yp, yd = ind.predict("FreeRun", y, u)
error = ind.score(yd, yp[:-1], measure="MSE")
assert np.isclose(error, 0), "something wrong with mimo_FreeRun/MSE"
error = ind.score(yd, yp[:-1], measure="MAPE")
assert np.isclose(error, 0), "something wrong with mimo_FreeRun/MAPE"

yp, yd = ind.predict("MShooting", 10, y, u)
error = ind.score(yd, yp, measure="MSE")
assert np.isclose(error, 0), "something wrong with miso_MShooting/MSE"
error = ind.score(yd, yp, measure="MAPE")
assert np.isclose(error, 0), "something wrong with miso_MShooting/MAPE"

