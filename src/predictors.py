# -*- coding: utf-8 -*-
"""
Created on Jun 2023

@author: Henrique Castro
"""
import multiprocessing
from concurrent.futures.thread import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from numba import njit
from concurrent.futures import ProcessPoolExecutor

def miso_OSA(ind, y, u):
    """
    Implements the One-Step_Ahead predictor for MISO models
    Arguments:
        ind = C_Individual object
        y   = 1-dimensional array with output data
        u   = n-dimensional array with input data
    """
    p = ind.makeRegressors(y, u)
    return np.dot(p, ind.theta), y[ind.lagMax + 1:]


def mimo_OSA(ind, y, u):
    """
    Implements the One-Step_Ahead predictor for MIMO models
    Arguments:
        ind = C_Individual object
        y   = n-dimensional array with output data
        u   = m-dimensional array with input data
    """
    P = ind.makeRegressors(y, u)
    # yp = []
    # for p, t in zip(P, np.array(ind.theta)):
    #     yp.append(np.dot(p, t))
    yp = [np.dot(p, t) for p, t in zip(P, np.array(ind._theta))]
    return np.array(yp).T, y[ind.lagMax + 1:]


def miso_FreeRun(ind, y0, u):
    """
    Implements the Free-Run predictor for MISO models
    Arguments:
        ind = C_Individual object
        y0  = 1-dimensional array with initial conditions
        u   = n-dimensional array with input data
    """
    y0 = y0.reshape(-1, 1)
    if len(u.shape) == 1:
        u = u.reshape(-1, 1)

    y = y0[:ind.lagMax + 1].reshape(-1, 1)

    for i in range(u.shape[0] - ind.lagMax):
        listV = [y[i:i + ind.lagMax + 1].reshape(-1, 1)]
        for v in u.T:
            listV.append(v[i:i + ind.lagMax + 1].reshape(-1, 1))

        p = [np.ones((ind.lagMax + 1))]

        for i in range(len(ind)):
            func = ind._funcs[i]
            out = func(*listV)
            p.append(out.reshape(-1))
        p = np.array(p).T[ind.lagMax:]
        y = np.vstack((y, np.dot(p, ind.theta)))
    return np.nan_to_num(y[:-1], nan=0), np.nan_to_num(y0, nan=0)


#def mimo_FreeRun(ind, y0, u):
#     """
#     Implements the Free-Run predictor for MIMO models
#     Arguments:
#         ind = C_Individual object
#         y0  = n-dimensional array with initial conditions
#         u   = m-dimensional array with input data
#     """
#     if len(u.shape) == 1:
#         u = u.reshape(-1, 1)
#
#     y = y0[:, :ind.lagMax + 1]
#
#     # Usando tqdm para adicionar a barra de progresso ao loop
#     for i in tqdm(range(u.shape[0] - ind.lagMax), desc="Processing iterations"):
#         listV = []
#         for v in y.T:
#             listV.append(v[i:i + ind.lagMax + 1].reshape(-1, 1))
#         for v in u.T:
#             listV.append(v[i:i + ind.lagMax + 1].reshape(-1, 1))
#
#         aux = []
#         for o in range(len(ind)):
#             p = [np.ones((ind.lagMax + 1))]
#             for i in range(len(ind[o])):
#                 func = ind._funcs[o][i]
#                 out = func(*listV)
#                 p.append(out.reshape(-1))
#             p = np.array(p).T[ind.lagMax:]
#             aux.append(np.dot(p, ind._theta[o].T))
#
#         y = np.vstack([y, np.array(aux).reshape(1, -1)])
#
#     return y[-(y0.shape[0] + 1):-1], y0

def compute_iteration(*args):
    from src.base import Individual
    i, ind, u, y = args
    listV = []
    for v in y.T:
        listV.append(v[i:i + ind.lagMax + 1].reshape(-1, 1))
    for v in u.T:
        listV.append(v[i:i + ind.lagMax + 1].reshape(-1, 1))

    aux = []
    for o in range(len(ind)):
        p = [np.ones((ind.lagMax + 1))]
        for j in range(len(ind[o])):
            func = ind._funcs[o][j]
            out = func(*listV)
            p.append(out.reshape(-1))
        p = np.array(p).T[ind.lagMax:]
        aux.append(np.dot(p, ind._theta[o].T))
    return np.array(aux).reshape(1, -1)

def mimo_FreeRun(ind, y0, u):
    if len(u.shape) == 1:
        u = u.reshape(-1, 1)

    y = y0[:, :ind.lagMax + 1]

    # results = Parallel(n_jobs=14, backend="threading")(
    #     delayed(compute_iteration)(i, ind, u, y) for i in
    #     tqdm(range(u.shape[0] - ind.lagMax), desc="Processing FreeRun iterations", total=(u.shape[0] - ind.lagMax))
    # )

    results = Parallel(n_jobs=14, backend="threading")(
        delayed(compute_iteration)(i, ind, u, y) for i in
        range(u.shape[0] - ind.lagMax)
    )

    for result in results:
        y = np.vstack([y, result])
    return y[-(y0.shape[0] + 1):-1], y0


def miso_MShooting(ind, k, y, u):
    """
    Implements the Multiple-Shooting predictor for MISO models
    Arguments:
        ind = C_Individual object
        k   = steps ahead prediction for each 'shooting'
        y   = 1-dimensional array with output data
        u   = n-dimensional array with input data
    """
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    if len(u.shape) == 1:
        u = u.reshape(-1, 1)

    n_batchs = int(np.floor(u.shape[0] / (ind.lagMax + 1 + k)))
    N = ind.lagMax + 1 + k
    newshape = (n_batchs, ind.lagMax + 1 + k, 1)
    listU = []
    for v in u.T:
        listU.append(np.resize(v, newshape))
    yk = np.resize(y, newshape)
    y0 = yk[:, :ind.lagMax + 1, :]
    for i in range(N - ind.lagMax - 1):
        p = []
        out = np.ones((n_batchs, 1, 1))
        p.append(out)
        for j in range(len(ind)):
            func = ind._funcs[j]
            listV = [y0[:, i:i + ind.lagMax + 1, :]]
            for v in listU:
                listV.append(v[:, i:i + ind.lagMax + 1, :])
            out = func(*listV)
            out = out[:, ind.lagMax:, :]
            p.append(out)
        p = np.concatenate(p, axis=2)
        y0 = np.concatenate((y0, np.dot(p, ind.theta)), axis=1)
    return y0.reshape(-1, 1), yk.reshape(-1, 1)


def mimo_MShooting(ind, k, y, u):
    """
    Implements the Multiple-Shooting predictor for MIMO models
    Arguments:
        ind = C_Individual object
        k   = steps ahead prediction for each 'shooting'
        y   = n-dimensional array with output data
        u   = m-dimensional array with input data
    """
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    if len(u.shape) == 1:
        u = u.reshape(-1, 1)

    # n_batchs = int(np.floor(u.shape[0] / (ind.lagMax + 1 + k)))
    n_batchs = u.shape[0] // (ind.lagMax + 1 + k)
    N = ind.lagMax + 1 + k
    newshape = (n_batchs, ind.lagMax + 1 + k, 1)

    listU = []
    for v in u.T:
        listU.append(np.resize(v, newshape))

    yk = np.resize(y, (n_batchs, ind.lagMax + 1 + k, y.shape[1]))

    y0 = yk[:, :ind.lagMax + 1, :]

    for i in range(N - ind.lagMax - 1):
        listV = []
        for v in y0.T:
            listV.append(v.T[:, i:i + ind.lagMax + 1].reshape(n_batchs, -1, 1))
        for v in listU:
            listV.append(v[:, i:i + ind.lagMax + 1, 0].reshape(n_batchs, -1, 1))

        aux = []
        for o in range(len(ind)):
            p = []
            out = np.ones((n_batchs, 1, 1))
            p.append(out)
            for j in range(len(ind[o])):
                func = ind._funcs[o][j]
                out = func(*listV)
                out = out[:, ind.lagMax:, :]
                p.append(out)
            p = np.concatenate(p, axis=2)
            # aux.append(np.dot(p, ind.theta.T[o]).reshape(-1, 1, 1))
            aux.append(np.dot(p, ind._theta[o].T).reshape(-1, 1, 1))

        y0 = np.concatenate((y0, np.concatenate(aux, axis=2)), axis=1)
    return np.nan_to_num(y0.reshape(-1, y.shape[1]), nan=0), np.nan_to_num(yk.reshape(-1, y.shape[1]), nan=0)
    # return y0.reshape(-1, y.shape[1]), yk.reshape(-1, y.shape[1])
