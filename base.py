# -*- coding: utf-8 -*-
"""
Created on Apr 2023

@author: Henrique Castro
"""

import pickle
from functools import partial
from abc import ABC, abstractmethod
from deap import gp, creator, base, tools
import operator
import numpy as np
import re
import warnings
from sklearn.metrics import mean_squared_error
from numba import njit, cuda

warnings.filterwarnings('ignore')
from predictors import miso_OSA, miso_FreeRun, miso_MShooting
from predictors import mimo_OSA, mimo_FreeRun, mimo_MShooting


#%% Element Class
def _roll(args, i):
    return np.roll(args, shift=i)


class Element(object):
    def __init__(self, weights=(-1,), delays=[1, 2, 3], nInputs=1, nOutputs=1, nTerms=10, maxHeight=5, mode="MISO"):
        self._mspset: gp.PrimitiveSet = None
        self._pset: gp.PrimitiveSet = None
        self._toolbox: base.Toolbox = None

        self._delays = delays
        self._nVar = nInputs + nOutputs
        self._weights = weights
        self._nTerms = nTerms
        self._nOutputs = nOutputs
        self._maxHeight = maxHeight
        self._mode = mode.upper()

        self.msPset
        self.pset

        creator.create("Program", gp.PrimitiveTree, fitness=None,
                       pset=self.pset)
        creator.create("FitnessMin", base.Fitness, weights=self._weights)

        if self._mode == "MISO":
            creator.create("Individual", IndividualMISO, fitness=creator.FitnessMin)
        elif self._mode == "MIMO":
            creator.create("Individual", IndividualMIMO, fitness=creator.FitnessMin)
        elif self._mode == "FIR":
            creator.create("Individual", IndividualFIR, fitness=creator.FitnessMin)
        else:
            raise Exception("Choose a mode between:\n" +
                            "MISO, MIMO, FIR")
        self.toolbox

    def getMode(self):
        return self._mode

    @property
    def pset(self):
        if self._pset == None:
            delays = [partial(_roll, i=i) for i in self._delays]
            self._pset = gp.PrimitiveSet("main", self._nVar)
            self._pset.addPrimitive(operator.mul, 2)
            #---set-one-step-ahead-pset---


            [self._pset.addPrimitive(roll, 1, name=f'q{i + 1}') for i, roll in enumerate(delays)]

        return self._pset

    @property
    def msPset(self):
        if self._mspset == None:
            delays = [partial(_roll, i=i) for i in self._delays]
            self._mspset = gp.PrimitiveSet("main", self._nVar)
            self._mspset.addPrimitive(operator.mul, 2)
            #---set-one-step-ahead-pset---

            [self._mspset.addPrimitive(roll, 1, name=f'q{i + 1}') for i, roll in enumerate(delays)]
        return self._mspset

    @property
    def toolbox(self):
        if self._toolbox == None:
            self._toolbox = base.Toolbox()
            self._toolbox.register("_expr", gp.genHalfAndHalf, pset=self._pset,
                                   min_=0, max_=self._maxHeight)
            self._toolbox.register("_program", tools.initIterate,
                                   creator.Program, self._toolbox._expr)
            if self._mode == "MISO":
                self._toolbox.register("individual", tools.initRepeat, creator.Individual,
                                       self._toolbox._program, self._nTerms)
            if self._mode == "MIMO":
                self._toolbox.register("_outputs", tools.initRepeat, list,
                                       self._toolbox._program, self._nTerms)
                self._toolbox.register("individual", tools.initRepeat, creator.Individual,
                                       self._toolbox._outputs, self._nOutputs)
            self._toolbox.register("population", tools.initRepeat, list,
                                   self._toolbox.individual)
        return self._toolbox

    def renameArguments(self, dictionary={'ARG0': 'y', 'ARG1': 'u'}):
        self._pset.renameArguments(**dictionary)
        self._mspset.renameArguments(**dictionary)

    def addPrimitive(self, *args):
        self._pset.addPrimitive(*args)
        self._mspset.addPrimitive(*args)

    def buildModelFromList(self, listString):
        model = creator.Individual()
        if self._mode == "MISO" or self._mode == "FIR":
            for string in listString:
                model.append(gp.PrimitiveTree.from_string(string, self.pset))
        if self._mode == "MIMO":
            for out in listString:
                # aux = []
                # for string in out:
                #     aux.append(gp.PrimitiveTree.from_string(string, self.pset))
                aux = [gp.PrimitiveTree.from_string(string, self.pset) for string in out]
                model.append(aux)
        return model

    def buildRandomModel(self):
        return self._toolbox.individual()

    def compileModel(self, model):
        if self._mode == 'MISO' or self._mode == "FIR":
            model._funcs = [gp.compile(tree, self.pset) for tree in model]
            model._msfuncs = [gp.compile(tree, self.msPset) for tree in model]
            self._setModelLagMax(model)

        if self._mode == 'MIMO':
            model._funcs = [[gp.compile(tree, self.pset) for tree in out] for out in model]
            model._msfuncs = [[gp.compile(tree, self.msPset) for tree in out] for out in model]

            self._setModelLagMax(model)

    def _setModelLagMax(self, model):
        def checkbranch(branch):
            if branch == []: return
            if branch[-1][2] == branch[-1][1]:
                del branch[-1]
                if branch == []: return
                branch[-1][2] += 1
                return checkbranch(branch)
            else:
                return

        def checkOut(output):
            treelags = []
            for tree in output:
                i = 0
                lagMax = 0
                branches = []
                count = 0
                while i < len(tree):
                    if re.search("q\d", tree[i].name):
                        count += int(tree[i].name[1:])
                    elif type(tree[i]) == gp.Primitive:
                        branches.append([count, tree[i].arity, 0])
                        count = 0
                    elif type(tree[i]) == gp.Terminal:
                        if branches == []:
                            lag = count
                            model._terminals += tree[i].value + '[i-%d] ' % (count + 1)
                        else:
                            branches[-1][2] += 1
                            lag = count + sum([item[0] for item in branches])
                            model._terminals += tree[i].value + '[i-%d] ' % (count + 1 + sum([item[0] for item in branches]))
                        if lag > lagMax:
                            lagMax = lag
                        count = 0
                        checkbranch(branches)
                    i += 1
                treelags.append(lagMax)
                model._terminals += '\n'
            model._terminals += '\n'
            return max(treelags)

        if self._mode == "MISO" or self._mode == "FIR":
            model.lagMax = checkOut(model)
        if self._mode == "MIMO":
            # i = 1
            # aux = []
            for i, _ in enumerate(model):
                model._terminals += 'Output %d:\n\n' % (i+1)
                # aux.append(checkOut(out))
                # i += 1
            aux = [checkOut(out) for out in model]
            model.lagMax = max(aux)

    #---save-load-file-function---------------------------------------------------------
    def save(self, filename, dictionary):
        with open(filename, 'wb') as f:
            pickle.dump(dictionary, f)
            f.close()

    def load(self, filename):
        with open(filename, 'rb') as f:
            o = pickle.load(f)
            f.close()
            return o


#%% Individual Abstract Class
class Individual(list):
    def __init__(self, data=[]):
        super().__init__(data)
        self._funcs = []
        self._msfuncs = []
        self._lagMax = None
        self._theta = []
        self._terminals = ''
        self._nTerms = 0

    @property
    def theta(self):
        if self._theta == []:
            raise Exception("Parameters \'theta\' are not defined!")
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = np.array(theta)

    @property
    def lagMax(self):
        return self._lagMax

    @lagMax.setter
    def lagMax(self, lag):
        self._lagMax = lag

    @abstractmethod
    def makeRegressors(self, y, u):
        pass

    @abstractmethod
    def predict(self, mode, *args):
        pass

    @abstractmethod
    def leastSquares(self, y, u):
        pass

    def _mape(self, yd, yp):
        N = yd.shape[0]
        return np.nan_to_num(100 * np.sum(np.abs(yd - yp)) / (N * np.abs(np.max(yd) - np.min(yd))), nan=np.inf)

    def score(self, yd, yp, mode="MSE"):
        if mode not in ["MSE", "MAPE"]:
            raise Exception("Choose a measure between:\n" +
                            "MSE, MAPE")
        if mode == "MSE":
            return mean_squared_error(yd, yp)
        if mode == "MAPE":
            return self._mape(yd, yp)

    @abstractmethod
    def model2List(self):
        pass


#%% MISO Element Class
@njit
def theta_miso(p, yd):
    return np.linalg.inv(p.T @ p) @ p.T @ yd


@njit
def theta_mimo(p, yd):
    return np.dot(np.dot(np.linalg.inv(np.dot(p.T, p)), p.T), yd)


@njit
def theta_fir(p, yd):
    return np.linalg.inv(p.T @ p) @ p.T @ yd


class IndividualMISO(Individual):

    def __init__(self, data=[]):
        super().__init__(data)

    def makeRegressors(self, y, u):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(u.shape) == 1:
            u = u.reshape(-1, 1)

        if y.shape[1] > 1:
            raise Exception('Wrong number of outputs. The algorithm is set',
                            'for a single output')

        listV = [y[:-1].reshape(-1, 1)]
        for v in u.T:
            listV.append(v[:-1].reshape(-1, 1))

        # p = np.ones((y.shape[0] - self.lagMax - 1, len(self) + 1))

        # for i in range(len(self)):
        #     func = self._funcs[i]
        #     out = func(*listV)
        #     p[:, i + 1] = out.reshape(-1)[self.lagMax:]

        p = np.array([np.ones(y.shape[0] - self.lagMax - 1) if i == 0 else
                      self._funcs[i - 1](*listV).reshape(-1)[self.lagMax:] for i in range(len(self) + 1)]).T
        return p

    def predict(self, mode="OSA", *args):
        if mode == "OSA":
            return miso_OSA(self, *args)
        if mode == "FreeRun":
            return miso_FreeRun(self, *args)
        if mode == "MShooting":
            return miso_MShooting(self, *args)
        else:
            raise Exception("Choose a mode between: OSA, FreeRun, MShooting")

    def leastSquares(self, y, u):
        '''
        The leastSquare(y,u) function implements the Least Squares method
        for parameter estimation.
        
        The arguments are the output y and the inputs u, in which each entry 
        must be in column formm.
        '''
        p = self.makeRegressors(y, u)
        if np.linalg.cond(p, -2) < 1e-10:
            raise np.linalg.LinAlgError(
                'Ill conditioned regressors matrix!')
        yd = y[self.lagMax + 1:]
        # self._theta = np.linalg.inv(p.T @ p) @ p.T @ yd
        # threadsperblock = 32
        # blockspergrid = (p.shape[0] + (threadsperblock - 1)) // threadsperblock
        # self._theta = my_function[blockspergrid, threadsperblock](p, yd)
        self._theta = theta_miso(p, yd)
        if len(self._theta.shape) == 1:
            self._theta = self._theta.reshape(-1, 1)
        return self._theta


    def __str__(self):
        string = ''.join('%s\n' * len(self)) % tuple([str(tree) for tree in self])
        return '1\n' + string

    def model2List(self):
        listString = []
        for tree in self:
            listString.append(str(tree))
        return listString


class IndividualMIMO(Individual):
    def __init__(self, data=[]):
        super().__init__(data)

    def makeRegressors(self, y, u):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(u.shape) == 1:
            u = u.reshape(-1, 1)

        if y.shape[1] == 1:
            raise Exception('Wrong number of outputs. The algorithm is set',
                            'for multiple outputs')
        listV = []
        for v in y.T:
            listV.append(v[:-1].reshape(-1, 1))
        for v in u.T:
            listV.append(v[:-1].reshape(-1, 1))

        P = []
        for o in range(len(self)):
            p = np.ones((y.shape[0] - self.lagMax - 1, len(self[o]) + 1))
            for i in range(len(self[o])):
                func = self._funcs[o][i]
                out = func(*listV)
                p[:, i + 1] = out.reshape(-1)[self.lagMax:]
            P.append(p)
        return P

    def predict(self, mode="OSA", *args):
        if mode == "OSA":
            return mimo_OSA(self, *args)
        if mode == "FreeRun":
            return mimo_FreeRun(self, *args)
        if mode == "MShooting":
            return mimo_MShooting(self, *args)
        else:
            raise Exception("Choose a mode between: OSA, FreeRun, MShooting")

    def leastSquares(self, y, u):
        '''
        The leastSquare(y,u) function implements the Least Squares method
        for parameter estimation.
        
        The arguments are the output y and the inputs u, in which each entry 
        must be in column formm.
        '''
        P = self.makeRegressors(y, u)
        for o in range(len(P)):
            p = P[o]
            if np.linalg.cond(p, -2) < 1e-10:
                raise np.linalg.LinAlgError(
                    'Ill conditioned regressors matrix!')
            yd = y[self.lagMax + 1:, o]
            # theta = np.dot(np.dot(np.linalg.inv(np.dot(p.T, p)), p.T), yd)
            theta = theta_mimo(p, yd)
            self._theta.append(theta)
        return np.array(self._theta).T

    def __str__(self):
        string = ''
        i = 1
        for out in self:
            string += 'Output %d:\n\n1\n' % (i)
            for tree in out:
                string += str(tree) + '\n'
            i += 1
            string += '\n'
        return string

    def model2List(self):
        listString = []
        for out in self:
            aux = []
            for tree in out:
                aux.append(str(tree))
            listString.append(aux)
        return listString
        # return [[str(tree) for tree in out] for out in self]



#%%
class IndividualFIR(Individual):

    def __init__(self, data=[]):
        super().__init__(data)

    def makeRegressors(self, y, u):
        if len(u.shape) == 1:
            u = u.reshape(-1, 1)

        listV = []
        for v in u.T:
            listV.append(v[:-1].reshape(-1, 1))

        p = np.ones((u.shape[0] - self.lagMax - 1, len(self) + 1))

        for i in range(len(self)):
            func = self._funcs[i]
            out = func(*listV)
            p[:, i + 1] = out.reshape(-1)[self.lagMax:]
        return p

    def predict(self, mode="OSA", *args):
        if mode == "OSA":
            return miso_OSA(self, *args)
        if mode == "FreeRun":
            return miso_FreeRun(self, *args)
        if mode == "MShooting":
            return miso_MShooting(self, *args)
        else:
            raise Exception("Choose a mode between: OSA, FreeRun, MShooting")

    def leastSquares(self, y, u):
        '''
        The leastSquare(y,u) function implements the Least Squares method
        for parameter estimation.
        
        The arguments are the output y and the inputs u, in which each entry 
        must be in column formm.
        '''
        p = self.makeRegressors(y, u)
        if np.linalg.cond(p, -2) < 1e-10:
            raise np.linalg.LinAlgError(
                'Ill conditioned regressors matrix!')
        yd = y[self.lagMax + 1:]
        # self._theta = np.linalg.inv(p.T @ p) @ p.T @ yd
        self._theta = theta_fir(p, yd)
        if len(self._theta.shape) == 1:
            self._theta = self._theta.reshape(-1, 1)
        return self._theta

    def __str__(self):
        string = ''.join('%s\n' * len(self)) % tuple([str(tree) for tree in self])
        return '1\n' + string

    def model2List(self):
        listString = []
        for tree in self:
            listString.append(str(tree))
        return listString

