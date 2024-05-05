# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:35:02 2023

@author: henrique
"""
import warnings

warnings.filterwarnings("ignore")

from abc import ABC, abstractmethod

from deap import tools, base

import numpy as np

import random

from copy import deepcopy

from mutations import *
from crossings import *


class Evolver(ABC):
    def __init__(self, element, evaluate, popSize):
        if evaluate == None:
            raise Exception('It needs an evaluation function!')
        if element == None:
            raise Exception('It needs an element module!')

        self._pop = []
        self._popSize = popSize

        self._hof = None
        self._element = element

        self._toolbox = base.Toolbox()
        self._toolbox.register("evaluate", evaluate)

        self._mutList = []
        self._crossList = []
        self._generation = 0

    def addMutation(self, mutation):
        self._mutList.append(mutation(self._element))

    def addCrossOver(self, crossover):
        self._crossList.append(crossover(self._element))

    def _delAttr(self, ind):
        try:
            del ind.fitness.values
            del ind.funcs
            del ind.kfuncs
            del ind.lagMax
        except AttributeError:
            pass

    @abstractmethod
    def _createStatistics(self):
        pass

    @abstractmethod
    def step(self):
        pass

    def getLogBook(self):
        return self._logbook

    def getHof(self):
        return self._hof

    def getPop(self):
        return self._pop

    def stream(self):
        print(self._logbook.stream)


class EvolDefault(Evolver):
    def __init__(self, element=None, evaluate=None, popSize=50, elitePerc=10, CXPB=0.8, MTPB=0.1):
        super().__init__(element, evaluate, popSize)

        self._elitePerc = elitePerc
        self._CXPB = CXPB
        self._MTPB = MTPB
        #---Setup--Statistics------------------------------------------------------
        self._stats = self._createStatistics()
        self._logbook = tools.Logbook()
        header = 'gen', 'evals', 'fitness'
        self._logbook.header = header
        self._logbook.chapters['fitness'].header = 'min', 'avg', 'max'

        self._hofSize = int(round(self._popSize * (self._elitePerc / 100)))
        self._hof = tools.HallOfFame(self._hofSize)

        self._toolbox.register("select", tools.selTournament, tournsize=2)

        self.addMutation(MutGPOneTree)
        self.addMutation(MutGPUniform)
        self.addMutation(MutGPReplace)

        self.addCrossOver(CrossHighUniform)
        self.addCrossOver(CrossLowUniform)

    def initPop(self, seed=[]):
        if len(seed) > self._popSize: raise Exception('Seed exceeds population size!')
        if seed == []:
            self._pop = self._element._toolbox.population(self._popSize)
        else:
            self._pop = self._element._toolbox.population(self._popSize - len(seed))
            self._pop += seed
        invalid_ind = [ind for ind in self._pop if not ind.fitness.valid]
        fitnesses = self._toolbox.map(self._toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        record = {'fitness': self._stats.compile(self._pop)}
        self._logbook.record(gen=self._generation, evals=len(invalid_ind), **record)
        # print(self._logbook)
        self._hof.update(self._pop)

    def _createStatistics(self):
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)
        return stats

    def step(self):
        if self._pop == []:
            raise Exception('Population must be initialized!')

        # offspring = list(map(deepcopy,
        #                      self._toolbox.select(self._pop, self._popSize - self._hofSize)))

        offspring = [deepcopy(ind) for ind in self._toolbox.select(self._pop, self._popSize - self._hofSize)]

        for i in range(0, len(offspring) - 1, 2):
            if np.random.random() < self._CXPB:
                cross = random.choice(self._crossList)
                offspring[i], offspring[i + 1] = cross.cross(offspring[i], offspring[i + 1])
                self._delAttr(offspring[i])
                self._delAttr(offspring[i + 1])

        for i in range(len(offspring)):
            if np.random.random() < self._MTPB:
                mut = random.choice(self._mutList)
                offspring[i], = mut.mutate(offspring[i])
                self._delAttr(offspring[i])

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = self._toolbox.map(self._toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        self._generation += 1

        self._pop = self._hof.items + offspring
        self._hof.update(self._pop)

        #---Record--Statistics-----------------------------------------------------
        record = {'fitness': self._stats.compile(self._pop)}

        self._logbook.record(gen=self._generation, evals=len(invalid_ind), **record)
