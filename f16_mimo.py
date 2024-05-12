# -*- coding: utf-8 -*-

import sys
from numba import njit
import pandas as pd

sys.path.insert(1, 'MGGP/')

import time
import numpy as np
# import matplotlib.pyplot as plt
from base import Element
from evolvers import EvolDefault
import multiprocessing

# def calculate_result(k):
#     if k >= 1:
#         model = 1 + y[k - 1] + u[k] * y[k - 1] + u[k]
#         print(f"y in k={k} is {y[k]} and de model is {model}. "
#               f"The MSE between y and model is {mean_squared_error(y[k], model)} ")
#     else:
#         print("K must be >= 1")


# def sys(N):
#     y1 = np.zeros((3 + N, 1))
#     u1 = np.random.normal(0, 1, (N + 3, 1))
#     y2 = np.zeros((3 + N, 1))
#     u2 = np.random.normal(0, 1, (N + 3, 1))
#     for i in range(3, N + 3):
#         y1[i] = 0.75 * y1[i - 2] + 0.25 * u1[i - 1] - 0.2 * y1[i - 2] * u2[i - 1] + 0.1 * y1[i - 3] * y2[i - 2]
#
#         y2[i] = 0.4 * u2[i - 1] ** 2 + 0.6 * y2[i - 1] - 0.45 * y2[i - 2] - 0.1 * y1[i - 1] - 0.2
#     y = np.concatenate([y1, y2], axis=1)
#     u = np.concatenate([u1, u2], axis=1)
#     return y[3:], u[3:]
#
#
# y, u = sys(1003)

df = pd.read_csv("F16GVT_Files/BenchmarkData/F16Data_FullMSine_Level1.csv").to_numpy()
y, u = (df[:, :2], df[:, 2:5])

element = Element(weights=(-1,), delays=[1, 2, 3], nInputs=3, nOutputs=2,
                  nTerms=3, maxHeight=5, mode="MIMO")
element.renameArguments({'ARG0': 'y1', 'ARG1': 'y2', 'ARG2': 'u1', 'ARG3': 'u2', 'ARG4': 'u3'})
# element.renameArguments({'ARG0': 'y1', 'ARG1': 'y2', 'ARG2': 'u1', 'ARG3': 'u2'})


k = 100


def evaluation(ind):
    try:
        element.compileModel(ind)
        theta_value = ind.leastSquares(y, u)
        ind._theta = list(theta_value)
        yp, yd = ind.predict("OSA", y, u)
        #yp, yd = ind.predict("MShooting", k, y, u)
        error = ind.score(yd, yp, "MSE")
        return error,
    except np.linalg.LinAlgError:
        return (np.inf,)


evolver = EvolDefault(element=element, evaluate=evaluation,
                      popSize=100, elitePerc=10, CXPB=0.8, MTPB=0.1)

if __name__ == "__main__":

    # pool = multiprocessing.Pool(4)  # using 4 processor cores
    # evolver._toolbox.register("map", pool.map)
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

    element.compileModel(model)
    theta_value = model.leastSquares(y, u)
    model._theta = list(theta_value)
    print(model)

    print(f"time: {round(end - init, 3)} seg")
    yp, yd = model.predict("OSA", y, u)

    # plt.figure(figsize=(10, 5))
    # plt.grid()
    # plt.title("Output 1")
    # plt.plot(y[:, 0])
    # plt.plot(yd[:, 0])
    # plt.savefig("output1.png")
    #
    # plt.figure(figsize=(10, 5))
    # plt.grid()
    # plt.title("Output 2")
    # plt.plot(y[:, 1])
    # plt.plot(yd[:, 1])
    # plt.savefig("output2.png")
