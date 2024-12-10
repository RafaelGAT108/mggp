# -*- coding: utf-8 -*-
"""
Created on Apr 25 2023

@author: Henrique Castro
"""

from abc import ABC, abstractmethod

from deap import gp, base

import random


class Crossing(ABC):
    def __init__(self, element):
        self._element = element
        self._toolbox = base.Toolbox()

    @abstractmethod
    def cross(self, ind1, ind2):
        pass

    def _gpConstraint(self, func, *args):
        clone = tuple(map(self._toolbox.clone, args))
        offspring = func(*args)
        for tree in offspring:
            if tree.height > self._element._maxHeight:
                return clone
        return offspring


class CrossLowOnePoint(Crossing):
    def __init__(self, element):
        super().__init__(element)

    def cross(self, ind1, ind2):
        if self._element._mode == 'MISO':
            idx = random.randint(0, len(ind1) - 1)
            ind1[idx], ind2[idx] = self._gpConstraint(gp.cxOnePoint, ind1[idx], ind2[idx])
            return ind1, ind2
        if self._element._mode == 'MIMO':
            idx = random.randint(0, len(ind1) - 1)
            idx2 = random.randint(0, len(ind1[0]) - 1)
            ind1[idx][idx2], ind2[idx][idx2] = self._gpConstraint(gp.cxOnePoint, ind1[idx][idx2], ind2[idx][idx2])
            return ind1, ind2


class CrossLowUniform(Crossing):
    def __init__(self, element):
        super().__init__(element)

    def cross(self, ind1, ind2):
        if self._element._mode == 'MISO':
            indpb = 0.5
            for i in range(len(ind1)):
                if random.random() < indpb:
                    ind1[i], ind2[i] = self._gpConstraint(gp.cxOnePoint, ind1[i], ind2[i])
            return ind1, ind2
        if self._element._mode == 'MIMO':
            indpb = 0.5
            for o in range(len(ind1)):
                for i in range(len(ind1[0])):
                    if random.random() < indpb:
                        ind1[o][i], ind2[o][i] = self._gpConstraint(gp.cxOnePoint, ind1[o][i], ind2[o][i])
            return ind1, ind2


class CrossHighOnePoint(Crossing):
    def __init__(self, element):
        super().__init__(element)

    def cross(self, ind1, ind2):
        if self._element._mode == 'MISO':
            idx = random.randint(0, len(ind1) - 1)
            aux = ind1[idx:]
            del ind1[idx:]
            ind1 += ind2[idx:]
            del ind2[idx:]
            ind2 += aux
            return ind1, ind2
        if self._element._mode == 'MIMO':
            for o in range(len(ind1)):
                idx = random.randint(1, len(ind1[o]) - 1)
                aux = ind1[o][idx:]
                del ind1[o][idx:]
                ind1[o] += ind2[o][idx:]
                del ind2[o][idx:]
                ind2[o] += aux
            return ind1, ind2


class CrossHighUniform(Crossing):
    def __init__(self, element):
        super().__init__(element)

    def cross(self, ind1, ind2):
        if self._element._mode == 'MISO':
            indpb = 0.5
            for i in range(len(ind1)):
                if random.random() < indpb:
                    aux = ind1[i]
                    ind1[i] = ind2[i]
                    ind2[i] = aux
            return ind1, ind2
        if self._element._mode == 'MIMO':
            for o in range(len(ind1)):
                indpb = 0.5
                for i in range(len(ind1[o])):
                    if random.random() < indpb:
                        aux = ind1[o][i]
                        ind1[o][i] = ind2[o][i]
                        ind2[o][i] = aux
            return ind1, ind2
